import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math

st.set_page_config(page_title="물류 배송 최적화 대시보드", page_icon="🚚", layout="wide")

st.title("🚚 라스트마일 물류 배송 최적화 (CVRP) 대시보드")
st.markdown("가장 늦게 배송을 마치는 차량의 완료 시간(Makespan)을 최소화하는 알고리즘 시뮬레이터입니다.")

# ----------------- 1. 차량 제약조건 설정 -----------------
st.sidebar.header("🚛 배송 차량 제약조건")
num_1ton = st.sidebar.number_input("1톤 탑차 수량", value=8, min_value=0)
cap_1ton = st.sidebar.number_input("1톤 탑차 적재량 (박스)", value=70)
num_2_5ton = st.sidebar.number_input("2.5톤 탑차 수량", value=4, min_value=0)
cap_2_5ton = st.sidebar.number_input("2.5톤 탑차 적재량 (박스)", value=150)

vehicle_capacities = [cap_1ton] * int(num_1ton) + [cap_2_5ton] * int(num_2_5ton)
num_vehicles = len(vehicle_capacities)

# ----------------- 2. 캡스톤 실제 노드 데이터 구성 -----------------
if 'node_data' not in st.session_state:
    # 캡스톤디자인 논문 데이터 참조
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
    
    # 기본 물동량 세팅 (1,000박스 기준 랜덤 배분)
    np.random.seed(42)
    demands = [0] + list(np.random.randint(15, 55, 30))
    demands = [0] + [int(d * (1000 / sum(demands[1:]))) for d in demands[1:]]
    demands[-1] += 1000 - sum(demands) # 1000박스 맞춤 오차 보정

    st.session_state.node_data = pd.DataFrame({
        '배송지명': node_names, '위도': lats, '경도': lons, '물동량(박스)': demands
    })

# ----------------- 3. 대시보드 UI 및 수정 가능한 표 -----------------
st.subheader("📍 거점별 배송지 및 물동량 데이터 (군포/안양/의왕 권역)")
st.caption("표 안의 데이터를 클릭하여 현장 수요에 맞게 직접 수정해 보세요. (에러 발생 요인 완벽 차단됨)")

edited_df = st.data_editor(st.session_state.node_data, num_rows="dynamic", use_container_width=True, height=350)

total_demand = edited_df['물동량(박스)'].sum()
total_capacity = sum(vehicle_capacities)

col1, col2 = st.columns(2)
col1.metric("총 할당 물동량", f"{total_demand} 박스")
col2.metric("총 차량 적재 가능량", f"{total_capacity} 박스")

if total_demand > total_capacity:
    st.error("⚠️ 총 물동량이 차량 최대 적재량을 초과했습니다.")

# ----------------- 4. OR-Tools 기반 Makespan 최적화 -----------------
if st.button("🚀 최적화 알고리즘 실행", type="primary"):
    with st.spinner('OR-Tools를 통해 차량별 최적 경로 및 완료 시간을 연산 중입니다...'):
        
        # 데이터프레임 인덱스 꼬임 방지를 위해 완벽한 리스트화
        names_list = edited_df['배송지명'].tolist()
        lats_list = edited_df['위도'].tolist()
        lons_list = edited_df['경도'].tolist()
        demands_list = edited_df['물동량(박스)'].tolist()
        n = len(names_list)
        
        # (1) 소요 시간(분) 매트릭스 계산: 거리 기반 + 하역 시간
        time_matrix = [[0]*n for _ in range(n)]
        dist_matrix = [[0]*n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    R = 6371.0
                    lat1, lon1, lat2, lon2 = map(math.radians, [lats_list[i], lons_list[i], lats_list[j], lons_list[j]])
                    a = math.sin((lat2 - lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1)/2)**2
                    dist_km = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                    dist_matrix[i][j] = dist_km
                    
                    # 이동 시간 산출 (평균 시속 30km 가정)
                    travel_time_min = (dist_km / 30.0) * 60
                    # 서비스 시간 (고정 10분 + 1분/박스)
                    service_time_min = 10 + (demands_list[j] * 1) if j != 0 else 0
                    
                    time_matrix[i][j] = int(travel_time_min + service_time_min)

        # (2) 라우팅 모델 구성
        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        # 시간(Time) 콜백 및 Dimension 설정 (Makespan 목표)
        def time_callback(from_index, to_index):
            return time_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
        
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # 최대 배송 완료 시간 최소화(Global Span Cost)
        routing.AddDimension(transit_callback_index, 0, 10000, True, 'Time')
        time_dimension = routing.GetDimensionOrDie('Time')
        time_dimension.SetGlobalSpanCostCoefficient(100)

        # 물동량(Capacity) 콜백
        def demand_callback(from_index):
            return demands_list[manager.IndexToNode(from_index)]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, vehicle_capacities, True, 'Capacity')

        # 파라미터 및 탐색 실행
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 5

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            st.success("✅ OR-Tools 해 탐색 완료!")
            
            # (3) Folium 맵 시각화 및 결과 집계
            m = folium.Map(location=[lats_list[0], lons_list[0]], zoom_start=13, tiles='CartoDB Positron')
            colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', 
                      '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff']
            
            max_makespan = 0
            
            for vehicle_id in range(num_vehicles):
                index = routing.Start(vehicle_id)
                route_coords = []
                route_load, route_time = 0, 0
                
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route_load += demands_list[node_index]
                    route_coords.append((lats_list[node_index], lons_list[node_index]))
                    
                    if node_index == 0:
                        folium.Marker([lats_list[node_index], lons_list[node_index]], 
                                      popup="🏢 군포 물류센터", icon=folium.Icon(color='black', icon='home')).add_to(m)
                    else:
                        folium.CircleMarker(
                            [lats_list[node_index], lons_list[node_index]], radius=6, 
                            popup=f"📦 {names_list[node_index]}<br>배송물량: {demands_list[node_index]}박스",
                            color=colors[vehicle_id % len(colors)], fill=True, fill_opacity=1
                        ).add_to(m)
                    
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_time += time_matrix[node_index][manager.IndexToNode(index)]
                
                # 물류센터 복귀 마감
                node_index = manager.IndexToNode(index)
                route_coords.append((lats_list[node_index], lons_list[node_index]))
                
                if len(route_coords) > 2:
                    max_makespan = max(max_makespan, route_time)
                    truck_type = "1톤" if vehicle_capacities[vehicle_id] <= 70 else "2.5톤"
                    folium.PolyLine(
                        route_coords, color=colors[vehicle_id % len(colors)], weight=4, opacity=0.8, 
                        popup=f"{truck_type} 차량 {vehicle_id+1} (총 소요: {route_time}분 / 적재: {route_load}박스)"
                    ).add_to(m)

            st.info(f"⏰ 가장 늦게 복귀하는 차량의 전체 배송 완료 시간 (Makespan): **{max_makespan} 분**")
            st_folium(m, width="1000", height=600, returned_objects=[])
        else:
            st.error("❌ 주어진 제약 조건 내에서 배송 경로를 찾지 못했습니다.")
