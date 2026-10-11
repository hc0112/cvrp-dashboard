import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math

st.set_page_config(page_title="물류 배송 최적화 대시보드", layout="wide")

st.title("🚚 군포·안양·의왕 권역 물류 배송 경로 최적화 (CVRP)")

# ----------------- 1. 차량 설정 -----------------
st.sidebar.header("🚛 배송 차량 제약조건")
num_1ton = st.sidebar.number_input("1톤 탑차 수량", value=8)
cap_1ton = st.sidebar.number_input("1톤 탑차 적재량 (박스)", value=70)
num_2ton = st.sidebar.number_input("2톤 탑차 수량", value=4)
cap_2ton = st.sidebar.number_input("2톤 탑차 적재량 (박스)", value=140)

vehicle_capacities = [cap_1ton] * int(num_1ton) + [cap_2ton] * int(num_2ton)
num_vehicles = len(vehicle_capacities)

# ----------------- 2. 배송지 및 물동량 설정 -----------------
st.subheader("📍 노드별 배송지 및 물동량 (성결대 제외)")
if 'node_data' not in st.session_state:
    np.random.seed(42)
    nodes = ['물류센터(0)'] + [f'배송지 단지({i})' for i in range(1, 31)]
    # 군포, 안양, 의왕 권역 반경 임의 설정
    lats = [37.3300] + list(np.random.uniform(37.3100, 37.3800, 30))
    lons = [126.9600] + list(np.random.uniform(126.9200, 126.9800, 30))
    # 30개 노드 총합이 정확히 1000박스가 되도록 분배
    demands_raw = np.random.randint(20, 45, 30)
    demands_adj = demands_raw * (1000 / demands_raw.sum())
    demands = [0] + [int(d) for d in demands_adj]
    demands[-1] += 1000 - sum(demands) # 오차 보정
    
    st.session_state.node_data = pd.DataFrame({
        '노드명': nodes, '위도': lats, '경도': lons, '물동량(박스)': demands
    })

edited_df = st.data_editor(st.session_state.node_data, num_rows="dynamic", use_container_width=True)

total_demand = edited_df['물동량(박스)'].sum()
total_capacity = sum(vehicle_capacities)
st.info(f"📦 현재 총 물동량: **{total_demand} 박스** / 🚛 총 차량 적재 가능량: **{total_capacity} 박스**")

if total_demand > total_capacity:
    st.warning("⚠️ 총 물동량이 차량의 최대 적재 가능량을 초과했습니다.")

# ----------------- 3. 거리 계산 및 최적화 실행 -----------------
if st.button("🚀 배송 경로 최적화 실행", type="primary"):
    with st.spinner('최적의 배송 경로를 탐색하고 있습니다...'):
        lats, lons = edited_df['위도'].tolist(), edited_df['경도'].tolist()
        n = len(lats)
        dist_matrix = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    R = 6371.0
                    lat1, lon1, lat2, lon2 = map(math.radians, [lats[i], lons[i], lats[j], lons[j]])
                    dlat, dlon = lat2 - lat1, lon2 - lon1
                    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                    dist_matrix[i][j] = int(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) * 1000)
        
        demands_list = edited_df['물동량(박스)'].tolist()
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

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.time_limit.seconds = 5

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            st.success("✅ 최적화 경로 탐색 완료!")
            m = folium.Map(location=[np.mean(lats), np.mean(lons)], zoom_start=13)
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                      'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']
            total_distance = 0
            
            for vehicle_id in range(num_vehicles):
                index = routing.Start(vehicle_id)
                route_coords, route_load, route_dist = [], 0, 0
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route_load += demands_list[node_index]
                    current_lat, current_lon = lats[node_index], lons[node_index]
                    route_coords.append((current_lat, current_lon))
                    
                    if node_index == 0:
                        folium.Marker([current_lat, current_lon], popup="물류센터", icon=folium.Icon(color='black', icon='home')).add_to(m)
                    else:
                        folium.Marker([current_lat, current_lon], popup=f"배송물량: {demands_list[node_index]}", icon=folium.Icon(color=colors[vehicle_id % len(colors)], icon='info-sign')).add_to(m)
                    
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_dist += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
                
                node_index = manager.IndexToNode(index)
                route_coords.append((lats[node_index], lons[node_index]))
                
                if len(route_coords) > 2:
                    truck_type = "1톤" if vehicle_capacities[vehicle_id] == 70 else "2톤"
                    folium.PolyLine(route_coords, color=colors[vehicle_id % len(colors)], weight=5, opacity=0.8, popup=f"{truck_type} 차량 {vehicle_id} (적재량: {route_load}박스)").add_to(m)
                    total_distance += route_dist

            st.metric(label="총 주행 거리", value=f"{total_distance / 1000:.2f} km")
            st_folium(m, width=1000, height=600)
        else:
            st.error("❌ 최적해를 찾을 수 없습니다.")