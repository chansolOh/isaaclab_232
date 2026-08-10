#!/bin/bash


env="Logistic_site"
python_file="../pre_grasp_point_sampler.py"
python_path="/home/cubox/.local/share/virtualenvs/IsaacLab-ov-BKEpu/bin/python"

data_dir="/nas/Dataset/Dataset_2025/test_data/$env/conf"
pre_grasp_dir="/nas/Dataset/Dataset_2025/test_data/$env/pre_grasp"

parent_dir=$(dirname "$PWD")




data_dir="/nas/Dataset/Dataset_2025/test_data/$env/conf"

cnt=0
pre_grasp_max=2
while true; do

    for file in "$pre_grasp_dir"/*.json; do
        name=$(basename "$file" .json)

        if [[ "$name" =~ ^[0-9]+$ ]]; then
            num=$((10#$name))  # 앞에 0 있어도 안전한 10진수 처리
            if (( num > pre_grasp_max )); then
                pre_grasp_max=$num
            fi
        fi
    done

    echo -e "\033[34m pre_grasp_max = $pre_grasp_max \033[0m"



    if [ -f "$data_dir/$(printf "%04d" "$((pre_grasp_max + 1))").json" ]; then
        $python_path $python_file --scene_num $((pre_grasp_max + 1)) --env $env --run_path $parent_dir
        cnt=0
    else
        echo "wait : $cnt"
        sleep 1
        cnt=$((cnt + 1))
    fi

    if [ $pre_grasp_max -eq 999 ]; then
        echo "1000개 모두 완료되었습니다."
        break
    fi


done
