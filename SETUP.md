# 팀 개발 환경 설정 (misys:forza_full + WSL)

교수님 이미지(`misys:forza_full`)의 `~/forza_ws`를 이 저장소로 바꿔서 쓰는 방법입니다.
코드는 WSL 디스크(`~/shared_dir`)에 두므로, 컨테이너를 지워도 사라지지 않습니다.

```
/home/misys/forza_ws  →(symlink)→  /home/misys/shared_dir/forza_ws  ⇄(mount)⇄  WSL /home/<사용자>/shared_dir/forza_ws
  코드가 찾는 경로                    컨테이너 안 실제 위치                        실제 저장 위치
```

설정 파일 곳곳에 `/home/misys/forza_ws` 경로가 하드코딩되어 있습니다(`.activate_forza.sh`, 글로벌 플래너의 `map_dir` 등).
그래서 저장소를 옮기는 대신, 이 경로에 symlink를 걸어 shared_dir의 저장소를 가리키게 합니다.

## 1. 처음 한 번

1. **WSL에서** 컨테이너를 만듭니다. 반드시 `dkrun_wsl.sh`로 실행해야 `~/shared_dir`가 연결됩니다.
   ```bash
   ./dkrun_wsl.sh misys:forza_full
   ```
2. **컨테이너 안에서** 저장소를 받고, 자기 브랜치를 만듭니다.
   ```bash
   git clone https://github.com/RE2031/aut_driving.git ~/shared_dir/forza_ws
   cd ~/shared_dir/forza_ws
   git switch -c <내 브랜치>          # 이미 GitHub에 있는 브랜치라면: git switch <브랜치>
   ```
3. 연결과 빌드를 합니다. 빌드는 몇 분 걸립니다.
   ```bash
   bash ~/shared_dir/forza_ws/setup_container.sh --build
   ```
4. git 사용자 정보를 등록합니다. 컨테이너마다 한 번씩 필요합니다.
   ```bash
   git config --global user.name  "<이름>"
   git config --global user.email "<GitHub 이메일>"
   ```

`setup_container.sh`가 하는 일:
1. 이미지에 원래 있던 `~/forza_ws`를 `~/forza_ws_base`로 보관합니다. 원본과 비교할 때 쓰면 됩니다.
2. `~/forza_ws` → `~/shared_dir/forza_ws` symlink를 만듭니다.
3. `race_stack/.activate_forza.sh`를 실행해서, Python이 이 저장소의 `f110_gym`을 쓰게 합니다.
4. `--build`를 붙이면 `colcon build`까지 합니다.

## 2. 평소 사용

- **같은 컨테이너를 다시 켤 때 (WSL에서)**
  ```bash
  docker start -ai <컨테이너ID>
  ```
  다른 터미널에서 같은 컨테이너에 들어갈 때:
  ```bash
  docker exec -it <컨테이너ID> bash
  ```
- **`dkrun_wsl.sh`로 새 컨테이너를 만들었을 때:** symlink와 Python 경로는 컨테이너 안에만 있으므로, 다시 연결해야 합니다. 빌드 결과는 shared_dir에 남아 있어 다시 빌드할 필요는 없습니다.
  ```bash
  bash ~/shared_dir/forza_ws/setup_container.sh
  ```
- **터미널마다 환경을 불러옵니다.**
  ```bash
  source /opt/ros/humble/setup.bash
  source ~/forza_ws/race_stack/install/setup.bash
  ```
  `~/f1tenth_ws`는 source하지 마세요. 같은 이름의 패키지가 많아서 엉뚱한 쪽이 실행될 수 있습니다.
- **설정이나 코드를 고친 뒤에는 다시 빌드합니다.** symlink 설치가 아니라서, 고친 패키지를 다시 빌드해야 반영됩니다.
  ```bash
  cd ~/forza_ws/race_stack && colcon build --packages-select <패키지>
  ```

## 3. 시뮬레이션 실행

터미널 3개에서 각각 실행합니다.
```bash
ros2 launch stack_master base_system_launch.xml map_name:=hyhall12 pose_source:=pf
ros2 launch stack_master head_to_head_launch.xml ctrl_algo:=PP
ros2 launch race_manager race_manager_launch.xml map_name:=hyhall12 seed:=2026 num_obstacles:=4 plot:=true
```

자주 헷갈리는 점:
- `ctrl_algo:=PP`는 C++ 제어기(`controller_cpp`)를 실행합니다. Python 버전(`controller/`)이 아닙니다.
- L1 파라미터는 `stack_master/config/global_params.yaml`의 `racecar_version`(현재 `NUC2`)을 따라 `stack_master/config/NUC2/l1_params.yaml`에서 읽습니다. `SIM/` 폴더가 아닙니다.
- 맵별 속도 상한은 `stack_master/maps/<맵>/speed_scaling.yaml`의 `global_limit`입니다.

## 4. Git 작업 흐름

```bash
git switch main && git pull                  # 최신 main 받기
git switch -c <브랜치>                        # 실험 단위 브랜치 (예: tune/l1-lookahead)
# ... 수정, 빌드, 랩타임 측정 ...
git add -A && git commit -m "무엇을 왜 바꿨는지"
git push -u origin <브랜치>                   # 처음 한 번만 -u, 이후로는 git push
```
- GitHub에서 `<브랜치>` → `main`으로 Pull Request를 만들고, 랩타임 결과를 함께 적습니다.
- `global_waypoints.json`처럼 생성되는 큰 파일은 담당자 한 명만 수정합니다. 충돌이 나면 손으로 합칠 수 없습니다.
- `shared_dir`는 WSL의 실제 폴더입니다. 컨테이너 안에서 `rm -rf ~/shared_dir/...`를 하면 원본이 지워집니다.
- 두 컨테이너에서 동시에 빌드하지 마세요. 같은 폴더를 공유하므로 서로 덮어씁니다.
