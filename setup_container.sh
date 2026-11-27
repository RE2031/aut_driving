#!/bin/bash
# 컨테이너(misys:forza_full) 안에서 실행: shared_dir에 받은 이 저장소를 원래 경로(~/forza_ws)에 연결한다.
# 여러 번 실행해도 안전하다. 새 컨테이너를 만들 때마다 한 번 실행하면 된다.
#
# 사용법:
#   bash ~/shared_dir/forza_ws/setup_container.sh           # 연결 + Python 경로 설정
#   bash ~/shared_dir/forza_ws/setup_container.sh --build   # 위 작업 + colcon build
set -e

REPO_DIR="$HOME/shared_dir/forza_ws"
LINK="$HOME/forza_ws"
BASE="$HOME/forza_ws_base"

if [ "$HOME" != "/home/misys" ]; then
    echo "[error] 컨테이너 안(HOME=/home/misys)에서 실행하세요. 현재 HOME=$HOME" >&2
    exit 1
fi
if [ ! -d "$REPO_DIR/race_stack" ]; then
    echo "[error] $REPO_DIR 에 저장소가 없습니다. 먼저 clone 하세요:" >&2
    echo "        git clone https://github.com/RE2031/aut_driving.git $REPO_DIR" >&2
    exit 1
fi

# 1. 이미지에 들어 있던 원본 forza_ws 를 forza_ws_base 로 보관 (처음 한 번만)
if [ -d "$LINK" ] && [ ! -L "$LINK" ]; then
    if [ -e "$BASE" ]; then
        echo "[error] $LINK 와 $BASE 가 둘 다 있습니다. 직접 확인하세요." >&2
        exit 1
    fi
    mv "$LINK" "$BASE"
    echo "[ok] 원본 forza_ws -> $BASE 로 보관"
fi

# 2. 하드코딩된 경로(/home/misys/forza_ws)가 shared_dir 의 저장소를 가리키게 연결
ln -sfn "$REPO_DIR" "$LINK"
echo "[ok] $LINK -> $REPO_DIR"

# 3. f110_gym 등 Python 경로를 이 저장소로 설정 (easy-install.pth)
cd "$LINK/race_stack"
bash .activate_forza.sh

# 4. (선택) 빌드
if [ "$1" = "--build" ]; then
    source /opt/ros/humble/setup.bash
    colcon build
fi

if [ -f "$LINK/race_stack/install/setup.bash" ]; then
    echo "[ok] 준비 완료. 새 터미널마다 실행: source /opt/ros/humble/setup.bash && source ~/forza_ws/race_stack/install/setup.bash"
else
    echo "[info] 아직 빌드되지 않았습니다. 실행: bash $REPO_DIR/setup_container.sh --build"
fi
