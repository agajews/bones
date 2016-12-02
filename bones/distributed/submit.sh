script_path="`dirname \"$0\"`"
bones_path="`dirname $\"script_path\"`/bones"
log_path=distributed_log.log
if (( "$#" < 3 ))
then
    echo "Too few parameters"
    echo "Expects at least <script.py> xx.xx.xx.xx yy.yy.yy.yy"
    exit 1
fi

script="$1"
for ip in "${@:2}"
do
    echo "Setting up $ip..."
    ssh -o StrictHostKeyChecking=no ubuntu@"$ip" < ${script_path}/gce_tf_setup.sh >> ${log_path} 2>&1
    ssh -o StrictHostKeyChecking=no ubuntu@"$ip" "rm -rf ~/bones" >> ${log_path} 2>&1
    scp "$script" ubuntu@"$ip":script.py >> ${log_path}
    rsync -zarv --include="*/" --include="*.py" --exclude="*" ${bones_path} ubuntu@"$ip":~ >> ${log_path}
done

function join_by { local IFS="$1"; shift; echo "$*"; }

worker_ips="$(join_by , "${@:3}")"
ssh ubuntu@"$2" "echo \"cd && python3 -u script.py --node_type=ps --ps_ip=$2 --worker_ips=$worker_ips\" > run.sh" >> ${log_path} 2>&1

i=0
for ip in "${@:3}"
do
    ssh -o StrictHostKeyChecking=no ubuntu@"$ip" "echo \"cd && python3 -u script.py --node_type=worker --idx=$i --ps_ip=$2 --worker_ips=$worker_ips\" > run.sh" >> ${log_path} 2>&1
    ((i++))
done

for ip in "${@:2}"
do
    echo "Rebooting $ip..."
    ssh -o StrictHostKeyChecking=no ubuntu@"$ip" 'sudo reboot' >> ${log_path} 2>&1
done

for ip in "${@:2}"
do
    while ! ssh ubuntu@"$ip" "echo 'Established connection!'" >> ${log_path} 2>&1
    do
        echo "Waiting 5 secs for response from $ip"
        sleep 5
    done
    ssh -o StrictHostKeyChecking=no ubuntu@"$ip" < ${script_path}/docker_start.sh >> ${log_path} 2>&1
done

ssh -o StrictHostKeyChecking=no ubuntu@"$3" 'docker logs tf-container -f'
