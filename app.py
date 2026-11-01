"""Motorcycle Safety Advisor — Streamlit research prototype.
Not an accident-probability or safety-certified system.
"""
from pathlib import Path
import json
import math
import time
import hashlib

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
import streamlit.components.v1 as components
import plotly.express as px
from catboost import Pool
from sklearn.neighbors import BallTree
import joblib

st.set_page_config(page_title='Motorcycle Safety Advisor', page_icon='🏍️', layout='wide')
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'assets' / 'catboost_final_v1_1.joblib'
HOTSPOT_PATH = ROOT / 'assets' / 'highway_hotspots_clean.csv'
R = 6371008.8
SIGMA = 250.0
RADIUS = 1500.0
SAMPLE_STEP = 25.0
RISK_MAP = {'No Accident':'Low', 'Moderate Accident':'Medium', 'Severe Accident':'High'}
CLASS_ORDER = ['No Accident', 'Moderate Accident', 'Severe Accident']

st.title('🏍️ Motorcycle Safety Advisor')
st.caption('Research prototype · CatBoost + SHAP + Highway Police blackspot proximity + OSRM')

@st.cache_resource(show_spinner=False)
def load_model(path, mtime):
    bundle = joblib.load(path)  # Only load files from trusted sources
    required = {'model', 'label_encoder', 'features', 'categorical_features'}
    if not required.issubset(bundle):
        raise ValueError('Incorrect model bundle schema')
    return bundle

@st.cache_data(show_spinner=False)
def load_hotspots(path, mtime):
    data = pd.read_csv(path)
    def column(candidates):
        lookup = {''.join(ch for ch in str(c).lower() if ch.isalnum()): c for c in data.columns}
        for name in candidates:
            key = ''.join(ch for ch in name.lower() if ch.isalnum())
            if key in lookup:
                return lookup[key]
        raise ValueError(f'Latitude/longitude columns not found: {list(data.columns)}')
    lat, lon = column(['Latitude','Lat','GPS_Latitude']), column(['Longitude','Lon','Lng','GPS_Longitude'])
    points = data[[lat,lon]].apply(pd.to_numeric, errors='coerce').dropna()
    points.columns = ['lat','lon']
    points = points[points.lat.between(20,27) & points.lon.between(87,93)].drop_duplicates()
    if points.empty: raise ValueError('No valid blackspot coordinates')
    return points.reset_index(drop=True)

missing = [str(p.relative_to(ROOT)) for p in [MODEL_PATH,HOTSPOT_PATH] if not p.is_file()]
if missing:
    st.error('Required research artifacts are missing: ' + ', '.join(missing))
    st.info('Copy the trained CatBoost bundle and cleaned blackspot CSV from Google Drive into the app/assets/ folder. See README.md.')
    st.stop()

try:
    bundle = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    hotspots = load_hotspots(str(HOTSPOT_PATH), HOTSPOT_PATH.stat().st_mtime_ns)
except Exception as exc:
    st.error(f'Could not load research artifacts: {exc}')
    st.stop()

model = bundle['model']
encoder = bundle['label_encoder']
features = list(bundle['features'])
cat_features = list(bundle['categorical_features'])
coords = hotspots[['lat','lon']].to_numpy(dtype=float)
tree = BallTree(np.radians(coords), metric='haversine')


def haversine(a, b):
    p1,p2=np.radians(np.asarray(a,dtype=float)),np.radians(np.asarray(b,dtype=float))
    d=p2-p1
    h=np.sin(d[0]/2)**2 + np.cos(p1[0])*np.cos(p2[0])*np.sin(d[1]/2)**2
    return float(2*R*np.arcsin(np.sqrt(np.clip(h,0,1))))


def sample_geometry(geojson):
    raw=np.asarray(geojson,dtype=float)
    if raw.ndim!=2 or raw.shape[1]!=2 or len(raw)<2: raise ValueError('Invalid route geometry')
    latlon=raw[:,::-1]
    phi=np.radians(latlon[:,0]); lam=np.radians(latlon[:,1]); dphi=np.diff(phi); dlam=np.diff(lam)
    h=np.sin(dphi/2)**2+np.cos(phi[:-1])*np.cos(phi[1:])*np.sin(dlam/2)**2
    lengths=2*R*np.arcsin(np.sqrt(np.clip(h,0,1)))
    cum=np.concatenate([[0.],np.cumsum(lengths)])
    mask=np.concatenate([[True],np.diff(cum)>1e-7])
    cum=cum[mask]; latlon=latlon[mask]
    if len(cum)<2 or cum[-1]<=0: raise ValueError('Zero-length geometry')
    dist=np.append(np.arange(0,cum[-1],SAMPLE_STEP),cum[-1])
    pts=np.column_stack([np.interp(dist,cum,latlon[:,0]),np.interp(dist,cum,latlon[:,1])])
    return pts,dist


def route_metrics(route):
    points, arclength=sample_geometry(route['geometry']['coordinates'])
    indices,distances=tree.query_radius(np.radians(points),r=RADIUS/R,return_distance=True)
    e=np.array([np.exp(-.5*(d*R/SIGMA)**2).sum() for d in distances],dtype=float)
    exposure=float(np.sum(.5*(e[1:]+e[:-1])*np.diff(arclength))/1000)
    near=set()
    for ix,d in zip(indices,distances): near.update(ix[d*R<=100].tolist())
    return {'Distance_km':route['distance']/1000,'Duration_min':route['duration']/60,
            'Integrated_Exposure_km':exposure,'Mean_Exposure':exposure*1000/arclength[-1],
            'Hotspots_100m':len(near)}


def osrm(points, alternatives, return_full=False):
    coord=';'.join(f'{lon:.7f},{lat:.7f}' for lat,lon in points)
    response=requests.get('https://router.project-osrm.org/route/v1/driving/'+coord,
        params={'alternatives':'true' if alternatives else 'false','overview':'full','geometries':'geojson'},
        headers={'User-Agent':'Motorcycle-Safety-Academic-Prototype/2.0'},timeout=40)
    response.raise_for_status(); data=response.json()
    if not isinstance(data,dict) or data.get('code')!='Ok': raise RuntimeError(f"OSRM response: {data.get('code') if isinstance(data,dict) else 'invalid'}")
    return data if return_full else data.get('routes',[])


def candidates_for(origin,destination,via_search):
    base=osrm([origin,destination],True)
    if not base: raise RuntimeError('No road route found')
    candidates=[(r,'OSRM') for r in base]
    if via_search:
        sample,d=sample_geometry(base[0]['geometry']['coordinates'])
        mid_idx=int(np.argmin(abs(d-d[-1]/2)))
        mid=sample[mid_idx]
        a=sample[max(0,mid_idx-5)]; b=sample[min(len(sample)-1,mid_idx+5)]
        dx=(b[1]-a[1])*math.cos(math.radians(mid[0])); dy=b[0]-a[0]
        norm=math.hypot(dx,dy)
        if norm>1e-12:
            for off in [-1.2,-.6,-.3,.3,.6,1.2]:
                waypoint=(float(mid[0]-dx/norm*off*1000/111195),
                          float(mid[1]+dy/norm*off*1000/(111195*math.cos(math.radians(mid[0])))))
                try:
                    time.sleep(1.1) # public demo endpoint, avoid rapid requests
                    response = osrm([origin,waypoint,destination],False,return_full=True)
                    snapped = None
                    osrm_waypoints = response.get('waypoints', [])
                    if len(osrm_waypoints) >= 3:
                        location = osrm_waypoints[1].get('location')
                        if isinstance(location, list) and len(location) == 2:
                            snapped = (float(location[1]), float(location[0]))
                    for r in response.get('routes', []):
                        if r['distance'] <= 1.8 * base[0]['distance']:
                            # Metadata stays attached to this route after deduplication.
                            enriched = dict(r)
                            enriched['_via_metadata'] = {
                                'requested': waypoint,
                                'snapped': snapped,
                                'offset_km': off,
                            }
                            candidates.append((enriched,'Via point'))
                except (requests.RequestException,ValueError,RuntimeError):
                    continue
    out=[]; seen=set()
    for route,source in candidates:
        xy=route['geometry']['coordinates']; start=(xy[0][1],xy[0][0]); end=(xy[-1][1],xy[-1][0])
        if haversine(start,origin)>300 or haversine(end,destination)>300: continue
        path,_=sample_geometry(xy)
        reduced=path[np.linspace(0,len(path)-1,80).astype(int)]
        key=tuple(np.round(reduced.flatten(),4))
        if key not in seen: seen.add(key); out.append((route,source))
    return out


def infer_behavior(values):
    x=pd.DataFrame([values],columns=features)
    age=float(x.loc[0,'Biker_Age']); exp=float(x.loc[0,'Riding_Experience'])
    if not (15<=age<=70 and 0<=exp<=30 and age-exp>=10):
        raise ValueError('Age/experience outside validated input constraints')
    for f in cat_features: x[f]=x[f].astype(str)
    pool=Pool(x,cat_features=cat_features)
    scores=np.asarray(model.predict_proba(pool))[0]
    pred_idx=int(np.argmax(scores)); pred=str(encoder.classes_[pred_idx])
    if pred not in RISK_MAP: raise ValueError(f'Unknown class: {pred}')
    sv=np.asarray(model.get_feature_importance(data=pool,type='ShapValues'))
    n=len(features); k=len(encoder.classes_)
    if sv.shape==(1,k,n+1): local=sv[0,pred_idx,:]
    elif sv.shape==(1,n+1,k): local=sv[0,:,pred_idx]
    else: raise ValueError(f'Unexpected SHAP dimensions: {sv.shape}')
    raw=np.asarray(model.predict(pool,prediction_type='RawFormulaVal')).reshape(-1)
    if not np.isclose(local[-1]+sum(local[:-1]),raw[pred_idx],atol=1e-5):
        raise RuntimeError('SHAP consistency check failed')
    contributions=pd.DataFrame({'Feature':features,'Contribution_to_raw_class_score':local[:-1]})
    contributions=contributions.iloc[np.argsort(abs(contributions.iloc[:,1].to_numpy()))[::-1]]
    scores_df=pd.DataFrame({'Dataset_class':encoder.classes_,'Model_class_score':scores})
    return pred,RISK_MAP[pred],scores_df,contributions


def rank_routes(records,mode):
    df=pd.DataFrame(records)
    arr=df[['Integrated_Exposure_km','Duration_min','Distance_km']].to_numpy(dtype=float)
    mini=arr.min(axis=0); span=arr.max(axis=0)-mini
    normalized=np.divide(arr-mini,span,out=np.zeros_like(arr),where=span>1e-12)
    df['Balanced_Cost']=normalized@np.array([.5,.3,.2])
    if mode=='Balanced': choice=df.sort_values(['Balanced_Cost','Duration_min','Distance_km']).iloc[0]
    elif mode=='Lowest Exposure':
        close=df[df.Integrated_Exposure_km<=df.Integrated_Exposure_km.min()+.005]
        choice=close.sort_values(['Duration_min','Distance_km']).iloc[0]
    elif mode=='Fastest': choice=df.sort_values(['Duration_min','Distance_km']).iloc[0]
    else: choice=df.sort_values(['Distance_km','Duration_min']).iloc[0]
    return df,choice

with st.sidebar:
    st.header('Rider information')
    age=st.number_input('Age (years)',min_value=15,max_value=70,value=28)
    exp=st.number_input('Riding experience (years)',min_value=0,max_value=30,value=5)
    helmet=st.selectbox('Helmet worn',['Yes','No'])
    license_=st.selectbox('Valid driving license',['Yes','No'])
    bike=st.selectbox('Bike condition',['New','Old'])
    road=st.selectbox('Road type (scenario assumption)',['City Road','Highway','Village Road'])
    weather=st.selectbox('Weather',['Clear','Foggy','Rainy'])
    daytime=st.selectbox('Time of day',['Morning','Noon','Afternoon','Evening','Night'])
    st.divider()
    st.header('Journey coordinates')
    st.caption('Bangladesh: latitude / longitude, decimal degrees')
    start_lat=st.number_input('Start latitude',value=23.503180006491085,format='%.8f')
    start_lon=st.number_input('Start longitude',value=90.9392392802157,format='%.8f')
    end_lat=st.number_input('End latitude',value=23.531652003412525,format='%.8f')
    end_lon=st.number_input('End longitude',value=91.08399838252356,format='%.8f')
    mode=st.selectbox('Route preference',['Balanced','Lowest Exposure','Fastest','Shortest'])
    via=st.checkbox('Explore additional via-point routes',value=True)
    go=st.button('Analyze rider and routes',type='primary',use_container_width=True)

st.info('Model class scores are not validated real-world accident probabilities. “Low / Medium / High” are display labels mapped from dataset classes. Road type is a user-provided scenario, not verified for a specific road.')

if not go:
    st.subheader('How to use')
    st.write('Enter rider characteristics and journey coordinates in the sidebar, then click **Analyze rider and routes**. The app displays CatBoost class scores and SHAP explanations alongside modeled hotspot exposure and candidate road routes.')
    st.caption('Research data coverage is limited to recorded Highway Police blackspots; zero observed exposure does not imply a safe route.')
    st.stop()

rider={'Biker_Age':age,'Riding_Experience':exp,'Wearing_Helmet':helmet,'Valid_Driving_License':license_,
       'Bike_Condition':bike,'Road_Type':road,'Weather':weather,'Time_of_Day':daytime}
try:
    predicted,risk,scores,shap_table=infer_behavior(rider)
except Exception as exc:
    st.error(f'Behavioral analysis failed: {exc}')
    st.stop()

st.subheader('1 · Rider behavioral assessment')
a,b,c=st.columns(3)
a.metric('Model-indicated category',risk)
b.metric('Predicted dataset class',predicted)
c.metric('Selected class score',f'{scores.loc[scores.Dataset_class==predicted,"Model_class_score"].iloc[0]*100:.2f}%')
st.dataframe(scores.sort_values('Model_class_score',ascending=False).assign(Model_class_score=lambda x:x.Model_class_score.round(6)),use_container_width=True,hide_index=True)

st.subheader('2 · SHAP explanation')
chart_df=shap_table.sort_values('Contribution_to_raw_class_score')
fig=px.bar(chart_df,x='Contribution_to_raw_class_score',y='Feature',orientation='h',title=f'Contributions to {predicted} raw score')
fig.add_vline(x=0,line_width=1,line_color='gray')
st.plotly_chart(fig,use_container_width=True)
st.caption('SHAP values explain the model’s predicted-class raw score (not percentage points of accident probability). Associations are not causal effects.')

origin=(float(start_lat),float(start_lon)); destination=(float(end_lat),float(end_lon))
if not all(math.isfinite(v) for v in (*origin,*destination)) or not all(20<=lat<=27 and 87<=lon<=93 for lat,lon in [origin,destination]):
    st.error('Please enter valid Bangladesh coordinates.'); st.stop()
if haversine(origin,destination)<50:
    st.error('Origin and destination must be at least 50 meters apart.'); st.stop()

st.subheader('3 · Route recommendation')
try:
    with st.spinner('Requesting road routes and evaluating recorded-blackspot exposure...'):
        candidate_routes=candidates_for(origin,destination,via)
        if not candidate_routes: raise ValueError('OSRM returned no eligible candidates')
        records=[{'Route':f'Route {i}','Source':source,**route_metrics(route)} for i,(route,source) in enumerate(candidate_routes,1)]
        route_df,chosen=rank_routes(records,mode)
except Exception as exc:
    st.warning(f'Route analysis unavailable: {exc}')
    st.caption('The behavioral ML result above is still valid as a model output; no route-safety conclusion can be drawn from this failed request.')
    st.stop()

r1,r2,r3=st.columns(3)
r1.metric(f'Recommended ({mode})',chosen['Route'])
r2.metric('Distance',f"{chosen['Distance_km']:.2f} km")
r3.metric('Travel duration (OSRM estimate)',f"{chosen['Duration_min']:.1f} min")
st.write(f"Integrated blackspot exposure: **{chosen['Integrated_Exposure_km']:.5f} exposure-km** · Hotspots within approx. 100 m: **{int(chosen['Hotspots_100m'])}**")
st.dataframe(route_df.round(5),hide_index=True,use_container_width=True)

# Display coordinates generated for every retained via-point route.
via_rows = []
for i, (route, source) in enumerate(candidate_routes, 1):
    via_meta = route.get('_via_metadata')
    if via_meta is None:
        continue
    requested = via_meta['requested']
    snapped = via_meta['snapped']
    via_rows.append({
        'Route': f'Route {i}',
        'Requested latitude': requested[0],
        'Requested longitude': requested[1],
        'OSRM snapped latitude': snapped[0] if snapped else None,
        'OSRM snapped longitude': snapped[1] if snapped else None,
        'Snap distance (m)': haversine(requested, snapped) if snapped else None,
        'Waypoint offset (km)': via_meta['offset_km'],
    })

st.subheader('Via-point coordinates')
if via_rows:
    via_table = pd.DataFrame(via_rows)
    st.dataframe(via_table, hide_index=True, use_container_width=True,
                 column_config={
                     'Requested latitude': st.column_config.NumberColumn(format='%.7f'),
                     'Requested longitude': st.column_config.NumberColumn(format='%.7f'),
                     'OSRM snapped latitude': st.column_config.NumberColumn(format='%.7f'),
                     'OSRM snapped longitude': st.column_config.NumberColumn(format='%.7f'),
                     'Snap distance (m)': st.column_config.NumberColumn(format='%.1f'),
                 })
    st.download_button('Download via-point coordinates (CSV)',
                       via_table.to_csv(index=False).encode('utf-8'),
                       file_name='route_via_points.csv', mime='text/csv')
    st.caption('Requested coordinates are the generated waypoints sent to OSRM. '
               'Snapped coordinates, when returned by OSRM, indicate the road-network position. '
               'Only via-points belonging to retained route candidates are listed.')
else:
    st.info('No retained via-point routes. Enable exploratory via-point routes or try another journey.')

st.subheader('4 · Map')
center=[(origin[0]+destination[0])/2,(origin[1]+destination[1])/2]
m=folium.Map(location=center,zoom_start=10,tiles='OpenStreetMap',control_scale=True)

# Consistent, unique color per route. Green is reserved for the selected route.
route_palette = [
    '#2563EB', '#F97316', '#9333EA', '#0891B2',
    '#B45309', '#DB2777', '#475569', '#7C3AED',
    '#0F766E', '#A16207'
]
route_colors = {}
for i in range(1, len(candidate_routes)+1):
    label=f'Route {i}'
    route_colors[label] = '#16A34A' if label == chosen['Route'] else route_palette[(i-1) % len(route_palette)]

# Clearly associate the map colors with route IDs and measured metrics.
st.markdown('**Route color legend** — green indicates the recommended route.')
legend_cols = st.columns(3)
for i, row in route_df.iterrows():
    label = str(row['Route'])
    route_color = route_colors[label]
    selected_tag = ' ✅ Recommended' if label == chosen['Route'] else ''
    description = (
        f"{float(row['Distance_km']):.2f} km · "
        f"{float(row['Duration_min']):.1f} min · "
        f"Exposure {float(row['Integrated_Exposure_km']):.3f}"
    )
    with legend_cols[i % 3]:
        st.markdown(
            f'<div style="border-left:8px solid {route_color};padding:8px 10px;margin-bottom:8px;'
            f'border-radius:4px;background:rgba(128,128,128,.08)">'
            f'<strong>{label}{selected_tag}</strong><div style="font-size:0.84rem">{description}</div></div>',
            unsafe_allow_html=True
        )

# Draw unselected routes first, selected route last so it is visible at overlaps.
order = [i for i in range(len(candidate_routes)) if f'Route {i+1}' != chosen['Route']]
order += [i for i in range(len(candidate_routes)) if f'Route {i+1}' == chosen['Route']]
for i in order:
    route, source = candidate_routes[i]
    latlon=[(point[1],point[0]) for point in route['geometry']['coordinates']]
    label=f'Route {i+1}'
    selected=(label == chosen['Route'])
    matching = route_df.loc[route_df['Route'] == label].iloc[0]
    details = (
        f"{label} {'(RECOMMENDED)' if selected else ''} | "
        f"{matching['Distance_km']:.2f} km | "
        f"{matching['Duration_min']:.1f} min | "
        f"Exposure {matching['Integrated_Exposure_km']:.3f}"
    )
    folium.PolyLine(
        latlon, color=route_colors[label],
        weight=8 if selected else 5,
        opacity=0.95 if selected else 0.8,
        tooltip=folium.Tooltip(details, sticky=True),
        popup=folium.Popup(details, max_width=350)
    ).add_to(m)

# Add requested and snapped via-point markers; selected-route marker is highlighted.
for i, (route, source) in enumerate(candidate_routes, 1):
    via_meta = route.get('_via_metadata')
    if via_meta is None:
        continue
    label = f'Route {i}'
    requested = via_meta['requested']
    snapped = via_meta['snapped']
    accent = route_colors[label]
    req_text = f'{requested[0]:.7f}, {requested[1]:.7f}'
    snap_text = (f'{snapped[0]:.7f}, {snapped[1]:.7f}' if snapped else 'Not returned')
    popup = (f'<b>{label} — Via-point</b><br>'
             f'Requested: {req_text}<br>OSRM snapped: {snap_text}')
    folium.Marker(
        location=requested,
        tooltip=f'{label} — requested waypoint',
        popup=folium.Popup(popup, max_width=350),
        icon=folium.DivIcon(html=(
            f'<div style="background:{accent};color:white;border:2px solid white;'
            f'border-radius:16px;min-width:42px;text-align:center;'
            f'font-weight:bold;padding:3px;box-shadow:0 1px 5px #444">'
            f'WP{i}</div>'
        ))
    ).add_to(m)
    if snapped is not None:
        folium.CircleMarker(
            location=snapped, radius=5, color=accent,
            fill=True, fill_color='white', fill_opacity=1,
            tooltip=f'{label} — OSRM snapped waypoint',
            popup=folium.Popup(popup, max_width=350)
        ).add_to(m)

folium.Marker(origin,tooltip='Origin',icon=folium.Icon(color='blue')).add_to(m)
folium.Marker(destination,tooltip='Destination',icon=folium.Icon(color='red')).add_to(m)
# Showing local blackspots within displayed route bounding box is illustrative.
all_route_pts=np.vstack([np.asarray(x[0]['geometry']['coordinates'],dtype=float) for x in candidate_routes])
west,east=all_route_pts[:,0].min(),all_route_pts[:,0].max()
south,north=all_route_pts[:,1].min(),all_route_pts[:,1].max()
subset=hotspots[hotspots.lon.between(west-.02,east+.02) & hotspots.lat.between(south-.02,north+.02)]
if len(subset)>200: subset=subset.iloc[:200]
for p in subset.itertuples(index=False):
    folium.CircleMarker((p.lat,p.lon),radius=4,color='crimson',fill=True,
                        fill_opacity=.7,tooltip='Recorded blackspot').add_to(m)
m.fit_bounds([(south,west),(north,east)])
components.html(m.get_root().render(),height=570,scrolling=False)
st.caption('Legend matches route colors. WP-number labels mark requested via-point coordinates; small outlined circles mark OSRM-snapped positions. Hover/click for coordinates. Red dots: recorded blackspots. Via-point routes are exploratory.')

st.subheader('5 · Interpretation')
st.write(f'**Rider:** {risk} model-indicated category ({predicted}). **Journey:** {chosen["Route"]} recommended using {mode.lower()} criteria. These are separate outcomes; the model does not estimate your individual crash probability on the selected route.')
st.warning('Research use only. Do not rely on this prototype as a navigation safety guarantee. Follow traffic law, current conditions, official route restrictions and real-time hazards.')
