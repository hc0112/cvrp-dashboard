import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import requests

st.set_page_config(page_title="물류 배송 최적화 대시보드", page_icon="🚛", layout="wide")

st.title("🚛 택배 물동량 기반 배송 경로 최적화 (CVRP)")
st.markdown("군포, 안양, 의왕 권역의 실제 도로망(OSRM API)을 반영한 다중 차량 배송 라우팅 시뮬레이션입니다.")

# ----------------- 1. 실제 도로 경로 탐색 함수 (OSRM API) -----------------
def get_osrm_route(coords):
    # coords: [(lat, lon), (lat, lon), ...]
    # OSRM은 경도(lon), 위도(lat) 순서로 요청해야 함
    coord_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
    url = f"http://router.project-osrm.org/route/v1/driving/{coord_str}?overview=full&geometries=geojson"
    try:
        response = requests.get(url, timeout=10).json()
        if response.get("code") == "Ok":
            # Folium은 위도(lat), 경도(lon) 순서로 그림
            route_coords = response["routes"][0]["geometry"]["coordinates"]
            return [[lat, lon] for lon, lat in route_coords]
    except:
        pass
    return coords # API 실패 시 직선으로 대체

# ----------------- 2. 차량 및 노드 초기 데이터 설정 -----------------
# 차량 제약조건 세팅 (1톤 8대, 2톤 4대)
num_1ton, cap_1ton = 8, 70
num_2ton, cap_2ton = 4, 140
vehicle_capacities = [cap_1ton] * num_1ton + [cap_2ton] * num_2ton
num_vehicles = len(vehicle_capacities)

if 'node_data' not in st.session_state:
    # 군포, 안양, 의왕 권역 실제 랜드마크 기반 가상 노드 생성 (성결대 제외)
    nodes = ['군포 복합물류터미널(센터)', '산본 래미안하이어스', '산본 주공11단지', '금정역 힐스테이트', '군포 당동푸르지오',
             '안양 래미안메가트리아', '평촌 아크로타워', '평촌 더샵센트럴시티', '안양 비산 래미안', '평촌 향촌현대',
             '안양 관양동 스마트타운', '안양 호계동 푸르지오', '명학역 반도유보라', '안양 인덕원마을', '안양 동편마을',
             '의왕 내손 e편한세상', '의왕 포일센트럴푸르지오', '의왕 서해그랑블', '의왕 백운밸리 효성', '의왕 오전동 푸르지오',
             '군포 부곡 휴먼시아', '군포 송정지구 금강', '군포 당정마을', '안양 석수 푸르지오', '안양 박달 한라비발디',
             '안양 안양천변 공구상가', '평촌 범계역 상가', '의왕 왕곡동 아파트', '의왕 청계마을', '의왕 삼동 푸르지오', '군포 군포역전시장']
    
    # 해당 지역 위경도 바운더리 내 임의 좌표 할당
    np.random.seed(42)
    lats = [37.3300] + list(np.random.uniform(37.3100, 37.4000, 30))
    lons = [126.9600] + list(np.random.uniform(126.9200, 126.9800, 30))
    
    # 30개 배송지의 총 물동량이 1000박스가 되도록 분배
    raw_demands = np.random.randint(15, 55, 30)
    demands = [0] + [int(d * (1000 / sum(raw_demands))) for d in raw_demands]
    demands[-1] += 1000 - sum(demands) # 나머지 오차 보정
    
    st.session_state.node_data = pd.DataFrame({
        '배송지명': nodes,
        '위도': [round(lat, 5) for lat in lats],
        '경도': [round(lon, 5) for lon in lons],
        '물동량(박스)': demands
    })

# ----------------- 3. 대시보드 탭 레이아웃 구성 -----------------
tab1, tab2 = st.tabs(["📝 데이터 및 제약조건 설정", "🗺️ 최적화 결과 및 지도 시각화"])

with tab1:
    st.markdown("### 📊 배송지 및 물동량 관리")
    st.caption("표 안의 데이터를 클릭하여 직접 수정할 수 있습니다. 변경된 데이터는 최적화 실행 시 즉각 반영됩니다.")
    
    # 편집 가능한 데이터프레임
    edited_df = st.data_editor(st.session_state.node_data, num_rows="dynamic", use_container_width=True, height=400)
    
    col1, col2 = st.columns(2)
    current_demand = edited_df['물동량(박스)'].sum()
    total_capacity = sum(vehicle_capacities)
    col1.metric(label="총 할당 물동량", value=f"{current_demand} 박스")
    col2.metric(label="총 차량 적재 가능량", value=f"{total_capacity} 박스")
    
    if current_demand > total_capacity:
        st.error("⚠️ 총 물동량이 차량 최대 적재량을 초과했습니다. 물량을 조정해주세요.")

with tab2:
    st.markdown("### 🚀 최적 경로 도출")
    
    if st.button("최적화 알고리즘 실행", type="primary", use_container_width=True):
        with st.spinner('OR-Tools 최적해 탐색 및 OSRM 도로 경로를 생성 중입니다. (약 5~10초 소요)...'):
            
            # 거리 행렬 연산 (OR-Tools 내부 계산용은 Haversine 사용하여 연산 속도 확보)
            lats, lons = edited_df['위도'].tolist(), edited_df['경도'].tolist()
            demands_list = edited_df['물동량(박스)'].tolist()
            n = len(lats)
            dist_matrix = [[0]*n for _ in range(n)]
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        R = 6371.0
                        lat1, lon1, lat2, lon2 = map(math.radians, [lats[i], lons[i], lats[j], lons[j]])
                        a = math.sin((lat2 - lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1)/2)**2
                        dist_matrix[i][j] = int(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) * 1000)

            # 라우팅 모델 구성
            manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
            routing = pywrapcp.RoutingModel(manager)

            def distance_callback(from_index, to_index):
                return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            def demand_callback(from_index):
                return demands_list[manager.IndexToNode(from_index)]
            
            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
            routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, vehicle_capacities, True, 'Capacity')

            # 파라미터 설정 및 솔루션 탐색
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            search_parameters.time_limit.seconds = 3

            solution = routing.SolveWithParameters(search_parameters)

            if solution:
                st.session_state['solution_found'] = True
                m = folium.Map(location=[np.mean(lats), np.mean(lons)], zoom_start=13, tiles='CartoDB Positron')
                colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', 
                          '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff']
                
                total_distance = 0
                total_load = 0
                used_vehicles = 0
                
                # 라우팅 결과 기반 OSRM 경로 시각화
                for vehicle_id in range(num_vehicles):
                    index = routing.Start(vehicle_id)
                    node_coords_for_api = []
                    route_load, route_dist = 0, 0
                    
                    while not routing.IsEnd(index):
                        node_index = manager.IndexToNode(index)
                        route_load += demands_list[node_index]
                        node_coords_for_api.append((lats[node_index], lons[node_index]))
                        
                        # 마커 추가
                        if node_index == 0:
                            folium.Marker([lats[node_index], lons[node_index]], popup="물류센터", icon=folium.Icon(color='black', icon='home')).add_to(m)
                        else:
                            folium.CircleMarker(
                                [lats[node_index], lons[node_index]], 
                                radius=6, 
                                popup=f"{edited_df.loc[node_index, '배송지명']}<br>배송물량: {demands_list[node_index]}박스",
                                color=colors[vehicle_id % len(colors)], fill=True, fill_opacity=1
                            ).add_to(m)
                        
                        previous_index = index
                        index = solution.Value(routing.NextVar(index))
                        route_dist += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
                    
                    # 복귀 지점(물류센터) 추가
                    node_index = manager.IndexToNode(index)
                    node_coords_for_api.append((lats[node_index], lons[node_index]))
                    
                    # 배송이 할당된 차량만 지도에 실제 도로망 경로 그리기
                    if len(node_coords_for_api) > 2:
                        used_vehicles += 1
                        total_distance += route_dist
                        total_load += route_load
                        
                        # OSRM API 호출하여 실제 주행경로 획득
                        real_route = get_osrm_route(node_coords_for_api)
                        truck_type = "1톤" if vehicle_capacities[vehicle_id] == 70 else "2톤"
                        
                        folium.PolyLine(
                            real_route, 
                            color=colors[vehicle_id % len(colors)], 
                            weight=4, opacity=0.8, 
                            popup=f"{truck_type} 차량 {vehicle_id} (적재량: {route_load}박스)"
                        ).add_to(m)

                # 최적화 맵을 session_state에 저장
                st.session_state['result_map'] = m
                st.session_state['kpi_distance'] = total_distance
                st.session_state['kpi_vehicles'] = used_vehicles
            else:
                st.error("❌ 최적해를 찾을 수 없습니다. 제약 조건을 확인해주세요.")

    # 지도가 session_state에 존재하면 렌더링 (버튼 누른 후 날아감 방지 핵심 옵션: returned_objects=[])
    if 'result_map' in st.session_state:
        st.success("✅ OSRM 도로망 기반 최적화 경로 탐색 완료!")
        col1, col2 = st.columns(2)
        col1.info(f"🚚 투입 차량 대수: **{st.session_state['kpi_vehicles']} 대** (총 12대 중)")
        col2.info(f"📏 총 예상 주행 거리: **{st.session_state['kpi_distance'] / 1000:.2f} km**")
        
        # 맵 렌더링 시 returned_objects=[] 옵션을 줘서 지도 클릭 시 새로고침 방지
        st_folium(st.session_state['result_map'], width="1000", height=600, returned_objects=[])
