#!/bin/bash


env="Home"
python_file="../Direct_RL_main.py"
python_path="/home/cubox/.local/share/virtualenvs/IsaacLab-ov-BKEpu/bin/python"

pre_grasp_dir="/nas/Dataset/Dataset_2025/test_data/$env/pre_grasp"
grasp_dir="/nas/Dataset/Dataset_2025/test_data/$env/output_grasp"

parent_dir=$(dirname "$PWD")


replay=false
file_check=false
time_th=300
cnt=800
while true; do



    if [ -f "$pre_grasp_dir/$(printf "%04d" "$cnt").json" ] && [ ! -f "$grasp_dir/$(printf "%04d" "$cnt").json" ]; then
        $python_path $python_file --scene_num $cnt --env $env --run_path $parent_dir &
        echo -e "\033[34m$cnt\033[0m"
        time_count=0
        while true; do
            if [ -f "$grasp_dir/$(printf "%04d" "$cnt").json" ]; then
                time_count=0
                echo -e "\033[34m$cnt\033[0m"
                echo -e "\033[34m파일생성 확인 완료\033[0m"

                file_check=true
                break


            else
                if ! pgrep -f $python_file > /dev/null; then
                    echo "replay"
                    sleep 5
                    replay=true
                    break
                fi
                time_count=$((time_count + 1))
                echo -e "\033[32m time_count = $time_count \033[0m"
                sleep 1
            fi
        
            if [ "$time_count" -ge "$time_th" ]; then
                replay=true
                echo "5분 동안 파일이 생성되지 않았습니다. 다시 시도합니다."
                break
            fi
        done
        
        if [ "$replay" = true ]; then
            kill -9 $(ps -ef | grep "$python_file" | grep -v "grep" | awk '{print $2}')
            replay=false
            sleep 5
            continue
        fi

        if [ "$file_check" = true ]; then
            file_check=false
            cnt=$((cnt + 1))
            sleep 5
            if pgrep -f $python_file > /dev/null; then
                kill -9 $(ps -ef | grep "$python_file" | grep -v "grep" | awk '{print $2}')
            fi
            continue
        fi



        if [ "$cnt" -eq 1000 ]; then
            echo "1000개 생성 완료"
            break
        fi
    elif [ ! -f "$pre_grasp_dir/$(printf "%04d" "$cnt").json" ]; then
        echo "wait : $cnt"
        sleep 1
    else
        cnt=$((cnt + 1))
    fi

done
