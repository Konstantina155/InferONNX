import os
import sys

if len(sys.argv) != 3:
    print("Usage: python3 run_all.py <partitions_folder> <number_of_runs>")
    exit(1)

have_certificates = True

os.chdir("..")
if not os.path.exists('occlum_workspace'):
    os.mkdir('occlum_workspace')
    os.mkdir('certificates')
    os.mkdir('unencrypted_models')
    os.mkdir('encrypted_models')
    have_certificates = False
os.chdir("occlum_workspace")

if not have_certificates:
    os.system("occlum init")
    os.system("mkdir image/bin/encrypted_models")
    os.system(f"openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout ../certificates/key.pem \
        -out ../certificates/cert.pem \
        -days 365 \
        -subj \"/C=US/ST=CA/L=SanFrancisco/O=MyCompany/CN=localhost\"")

os.system(f"cp ../certificates/* image/bin")

inferONNX_path = os.getcwd() + "/../InferONNX"
path_to_scripts = inferONNX_path + "/scripts"
tag_tls_server = inferONNX_path + '/src/server_with_tls/tag_file.txt'
if not os.path.exists(tag_tls_server):
    with open(tag_tls_server, "w") as f:
        f.write("")

tag_no_tls_server = inferONNX_path + '/src/server_without_tls/tag_file.txt'
if not os.path.exists(tag_no_tls_server):
    with open(tag_no_tls_server, "w") as f:
        f.write("")

'''
os.system(f"python3 {path_to_scripts}/run_tests_occlum.py on_disk entire {number_of_runs} {inferONNX_path}")
os.system(f"python3 {path_to_scripts}/run_tests_occlum.py on_disk partitions {number_of_runs} {inferONNX_path}")
os.system(f"python3 {path_to_scripts}/run_tests_occlum.py memory_only entire {number_of_runs} {inferONNX_path}")
os.system(f"python3 {path_to_scripts}/run_tests_cpu.py tls_memory_only {number_of_runs} {inferONNX_path}")
os.system(f"python3 {path_to_scripts}/run_tests_cpu.py on_disk {number_of_runs} {inferONNX_path}")
os.system(f"python3 {path_to_scripts}/run_tests_cpu.py memory_only {number_of_runs} {inferONNX_path}")
'''

os.system(f"python3 {path_to_scripts}/create_plots.py {inferONNX_path} {sys.argv[1]} {sys.argv[2]}")
# command to generate the two plots and the figure

os.remove(tag_no_tls_server)
os.remove(tag_tls_server)
os.chdir("../InferONNX")
os.system('rm -rf ../occlum_workspace')
os.system('rm -rf ../certificates')
os.system('rm -rf ../unencrypted_models')
os.system('rm -rf ../encrypted_models')