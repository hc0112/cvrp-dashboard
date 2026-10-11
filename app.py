import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import requests

st.set_page_config(page_title="물류 배송 최적화 대시보드", page_icon="🚚", layout="wide")

st.title("🚚 군포·의왕 권역 배송 최적화 (CVRP) 시뮬레이터")

# ----------------- 1. 실제 도로망 경로 탐색 (OSRM API) -----------------
def get_osrm_route(coords):
    coord_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
    url = f"http://router.project-osrm.org/route/v1/driving/{coord_str}?overview=full&geometries=geojson"
    try:
        response = requests.get(url, timeout=10).json()
        if response.get("code") == "Ok":
            route_coords = response["routes"][0]["geometry"]["coordinates"]
            return [[lat, lon] for lon, lat in route_coords], response["routes"][0]["distance"] / 1000
    except:
        pass
    return coords, 0 # API 실패 시 직선 대체

# ----------------- 2. 차량 제약조건 -----------------
st.sidebar.header("🚛 배송 차량 제약조건")
num_1ton = st.sidebar.number_input("1톤 탑차 수량", value=8, min_value=0)
cap_1ton = st.sidebar.number_input("1톤 탑차 적재량 (박스)", value=70)
num_2_5ton = st.sidebar.number_input("2.5톤 탑차 수량", value=4, min_value=0)
cap_2_5ton = st.sidebar.number_input("2.5톤 탑차 적재량 (박스)", value=150)

vehicle_capacities = [cap_1ton] * int(num_1ton) + [cap_2_5ton] * int(num_2_5ton)
num_vehicles = len(vehicle_capacities)

# ----------------- 3. 거점 및 물동량 데이터 -----------------
if 'node_data' not in st.session_state:
    node_names = [
        "군포 복합물류터미널", "용호마을 e편한세상", "래미안하이어스", "금정역 e편한세상 2차", 
        "힐스테이트 금정역", "래미안 안양 메가트리아", "평촌 자이아이파크", "평촌더샵아이파크", 
        "호계2차현대홈타운", "평촌 어바인퍼스트", "내손 e편한세상", "의왕더샵캐슬", 
        "백운호수해링턴 2단지", "산본 주공 11단지", "래미안 수리", "부곡지구 휴먼시아", 
        "송정지구 금강펜테리움", "당동 주공 4단지", "비산 삼성래미안", "관양 동편마을 4단지", 
        "인덕원 푸르지오 엘센트로", "평촌 한가람단지", "박달 한라비발디", "석수 두산위브", 
        "포일 자이", "청계 휴먼시아", "의왕 서밋포레", "오전 해모로", "고천 서해그랑블", 
        "장안지구 파크푸르지오", "왕송호수 푸르지오"
    ]
    
    lats = [37.329, 37.345, 37.362, 37.368, 37.370, 37.395, 37.390, 37.382, 37.374, 37.378,
            37.385, 37.363, 37.371, 37.360, 37.350, 37.315, 37.320, 37.340, 37.405, 37.405,
            37.398, 37.395, 37.400, 37.415, 37.390, 37.395, 37.355, 37.365, 37.345, 37.335, 37.310]
            
    lons = [126.918, 126.925, 126.936, 126.944, 126.942, 126.925, 126.935, 126.946, 126.958, 126.950,
            126.975, 126.963, 127.006, 126.920, 126.915, 126.925, 126.905, 126.945, 126.940, 126.965,
            126.975, 126.955, 126.905, 126.910, 126.985, 127.000, 126.975, 126.975, 126.970, 126.955, 126.945]
    
    np.random.seed(42)
    demands = [0] + list(np.random.randint(15, 55, 30))
    demands = [0] + [int(d * (1000 / sum(demands[1:]))) for d in demands[1:]]
    demands[-1] += 1000 - sum(demands)

    st.session_state.node_data = pd.DataFrame({
        '배송지명': node_names, '위도': lats, '경도': lons, '물동량(박스)': demands
    })

st.subheader("📍 데이터 관리 (수정 가능)")
edited_df = st.data_editor(st.session_state.node_data, num_rows="dynamic", use_container_width=True, height=250)

# ----------------- 4. 최적화 및 시각화 -----------------
if st.button("🚀 배송 라우팅 최적화 실행 (OSRM 연동)", type="primary"):
    with st.spinner('OR-Tools 연산 및 OSRM 실제 도로망 맵핑 중입니다...'):
        
        names_list = edited_df['배송지명'].tolist()
        lats_list = edited_df['위도'].tolist()
        lons_list = edited_df['경도'].tolist()
        demands_list = edited_df['물동량(박스)'].tolist()
        n = len(names_list)
        
        time_matrix = [[0]*n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    R = 6371.0
                    lat1, lon1, lat2, lon2 = map(math.radians, [lats_list[i], lons_list[i], lats_list[j], lons_list[j]])
                    a = math.sin((lat2 - lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1)/2)**2
                    dist_km = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                    
                    travel_time_min = (dist_km / 30.0) * 60
                    service_time_min = 10 + (demands_list[j] * 1) if j != 0 else 0
                    time_matrix[i][j] = int(travel_time_min + service_time_min)

        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index, to_index):
            return time_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        routing.AddDimension(transit_callback_index, 0, 10000, True, 'Time')
        time_dimension = routing.GetDimensionOrDie('Time')
        time_dimension.SetGlobalSpanCostCoefficient(100)

        def demand_callback(from_index):
            return demands_list[manager.IndexToNode(from_index)]
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, vehicle_capacities, True, 'Capacity')

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 5

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            st.success("✅ 해 탐색 및 맵핑 완료!")
            
            # API KEY 오류를 해결한 OpenStreetMap 사용
            m = folium.Map(location=[lats_list[0], lons_list[0]], zoom_start=13, tiles='OpenStreetMap')
            colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', 
                      '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff']
            
            max_makespan = 0
            route_results = [] # 엑셀 형식의 결과 저장을 위한 리스트
            
            for vehicle_id in range(num_vehicles):
                index = routing.Start(vehicle_id)
                node_coords_for_api = []
                route_names = []
                route_load, route_time = 0, 0
                
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route_load += demands_list[node_index]
                    node_coords_for_api.append((lats_list[node_index], lons_list[node_index]))
                    route_names.append(names_list[node_index])
                    
                    if node_index == 0:
                        folium.Marker([lats_list[node_index], lons_list[node_index]], 
                                      icon=folium.Icon(color='black', icon='home')).add_to(m)
                    else:
                        folium.CircleMarker(
                            [lats_list[node_index], lons_list[node_index]], radius=6, 
                            popup=f"{names_list[node_index]}<br>{demands_list[node_index]}박스",
                            color=colors[vehicle_id % len(colors)], fill=True, fill_opacity=1
                        ).add_to(m)
                    
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_time += time_matrix[node_index][manager.IndexToNode(index)]
                
                node_index = manager.IndexToNode(index)
                node_coords_for_api.append((lats_list[node_index], lons_list[node_index]))
                route_names.append(names_list[node_index])
                
                if len(node_coords_for_api) > 2:
                    max_makespan = max(max_makespan, route_time)
                    truck_type = "1톤" if vehicle_capacities[vehicle_id] <= 70 else "2.5톤"
                    
                    # OSRM 도로망 경로 호출
                    real_route_coords, real_dist_km = get_osrm_route(node_coords_for_api)
                    
                    folium.PolyLine(
                        real_route_coords, color=colors[vehicle_id % len(colors)], weight=5, opacity=0.8, 
                    ).add_to(m)
                    
                    # 상세 결과 표를 위한 데이터 수집
                    route_results.append({
                        "차량 번호": f"{truck_type} ({vehicle_id+1}호차)",
                        "운행 경로": " ➔ ".join(route_names),
                        "주행 거리(km)": round(real_dist_km, 2),
                        "소요 시간(분)": route_time,
                        "적재량(박스)": route_load
                    })

            # 시각화 출력 파트
            col1, col2 = st.columns([1, 1])
            with col1:
                st.info(f"⏰ 최대 배송 완료 시간 (Makespan): **{max_makespan} 분**")
            with col2:
                st.info(f"🚚 실제 운행 투입 차량: **{len(route_results)} 대**")

            st_folium(m, width="1000", height=500, returned_objects=[])
            
            # 차량별 상세 결과 데이터프레임 (엑셀 형식)
            st.markdown("### 📊 차량별 상세 운행 결과")
            results_df = pd.DataFrame(route_results)
            st.dataframe(results_df, use_container_width=True)
            
        else:
            st.error("❌ 주어진 조건으로 경로를 찾지 못했습니다. 차량을 늘리거나 물량을 줄여주세요.")
