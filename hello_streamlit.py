import streamlit as st
import random
import time
import numpy as np

# 초기 설정
if 'base_sudoku' not in st.session_state:
    st.session_state.base_sudoku = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [4, 5, 6, 7, 8, 9, 1, 2, 3],
        [7, 8, 9, 1, 2, 3, 4, 5, 6],
        [2, 3, 1, 8, 9, 7, 5, 6, 4],
        [5, 6, 4, 2, 3, 1, 8, 9, 7],
        [8, 9, 7, 5, 6, 4, 2, 3, 1],
        [3, 1, 2, 6, 4, 5, 9, 7, 8],
        [6, 4, 5, 9, 7, 8, 3, 1, 2],
        [9, 7, 8, 3, 1, 2, 6, 4, 5]
    ]
if 'AVal' not in st.session_state:
    st.session_state.AVal = [[0 for _ in range(9)] for _ in range(9)]
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0
if 'elapsed_time' not in st.session_state:
    st.session_state.elapsed_time = 0
if 'rankings' not in st.session_state:
    st.session_state.rankings = []

# 타이머 시작
def start_timer():
    st.session_state.start_time = time.time()

# 타이머 정지
def stop_timer():
    st.session_state.elapsed_time = time.time() - st.session_state.start_time

# 셔플 버튼 동작
def shuffle_sudoku():
    random_mapping = random.sample(range(1, 10), 9)
    for i in range(9):
        for j in range(9):
            original_value = st.session_state.base_sudoku[i][j]
            st.session_state.AVal[i][j] = random_mapping[original_value - 1]
    
    # 셀을 30% 확률로 빈칸으로 만들기
    for i in range(9):
        for j in range(9):
            if random.random() < 0.3:
                st.session_state.AVal[i][j] = 0
    
    start_timer()

# 사용자가 입력한 값으로 AVal 갱신
def update_grid():
    for i in range(9):
        for j in range(9):
            try:
                value = int(st.session_state[f"cell_{i}_{j}"])
                st.session_state.AVal[i][j] = value
            except:
                st.session_state.AVal[i][j] = 0

# 해법 확인
def check_solution():
    for i in range(9):
        if len(set(st.session_state.AVal[i])) != 9:
            st.write(f"행 {i+1}에 오류가 있습니다.")
            return False
    for j in range(9):
        column = [st.session_state.AVal[i][j] for i in range(9)]
        if len(set(column)) != 9:
            st.write(f"열 {j+1}에 오류가 있습니다.")
            return False
    for box_row in range(3):
        for box_col in range(3):
            box = [st.session_state.AVal[box_row*3 + i][box_col*3 + j] for i in range(3) for j in range(3)]
            if len(set(box)) != 9:
                st.write(f"{box_row*3 + 1},{box_col*3 + 1}의 3x3 그리드에 오류가 있습니다.")
                return False
    return True

# Streamlit UI 구성
st.title("Sudoku Solver with Streamlit")
st.write("빈칸에 숫자를 입력하고 'Finish' 버튼을 클릭하세요!")

# CSS 스타일을 이용한 3x3 구역 강조
cell_css = """
<style>
.sudoku-cell {
    width: 40px;
    height: 40px;
    text-align: center;
    font-size: 20px;
}
.border-strong {
    border: 2px solid black;
}
.border-right {
    border-right: 3px solid black;
}
.border-bottom {
    border-bottom: 3px solid black;
}
</style>
"""

# CSS 스타일을 HTML로 적용
st.markdown(cell_css, unsafe_allow_html=True)

# Sudoku 그리드 표시 및 입력
grid_html = "<div style='display: grid; grid-template-columns: repeat(9, 40px);'>"
for i in range(9):
    for j in range(9):
        cell_key = f"cell_{i}_{j}"
        cell_value = "" if st.session_state.AVal[i][j] == 0 else str(st.session_state.AVal[i][j])
        
        # 각 셀의 경계 스타일 설정
        border_class = "sudoku-cell "
        if j % 3 == 2 and j != 8:
            border_class += "border-right "
        if i % 3 == 2 and i != 8:
            border_class += "border-bottom "
        grid_html += f"<input type='text' id='{cell_key}' value='{cell_value}' class='{border_class}' maxlength='1' oninput='this.value=this.value.replace(/[^0-9]/g,\"\");' />"

grid_html += "</div>"

# Sudoku 그리드를 HTML로 표시
st.markdown(grid_html, unsafe_allow_html=True)

# 버튼 UI
if st.button("Shuffle"):
    shuffle_sudoku()
if st.button("Finish"):
    stop_timer()
    if check_solution():
        st.write(f"축하합니다! 퍼즐을 {st.session_state.elapsed_time:.2f}초 만에 완료했습니다.")
        name = st.text_input("이름을 입력하세요:", "")
        if name:
            st.session_state.rankings.append((name, st.session_state.elapsed_time))
            st.session_state.rankings.sort(key=lambda x: x[1])
    else:
        st.write("오류가 있습니다. 다시 확인해주세요.")

# 랭킹 표시
st.subheader("랭킹")
for rank, (name, elapsed) in enumerate(st.session_state.rankings, 1):
    st.write(f"{rank}. {name} - {elapsed:.2f} 초")
