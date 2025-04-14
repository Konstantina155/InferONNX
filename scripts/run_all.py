import os

# TODO: modify the path to certificates in ssl_client to ~/

path_to_occlum = '/hdd/papafrkon/InferONNX/scripts' # TODO: change it to home dir
os.chdir(path_to_occlum) # os.path.expanduser("~"))
have_certificates = True
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
    -keyout {path_to_occlum}/certificates/key.pem \
    -out {path_to_occlum}/certificates/cert.pem \
    -days 365 \
    -subj \"/C=US/ST=CA/L=SanFrancisco/O=MyCompany/CN=localhost\"")

os.system(f"cp {path_to_occlum}/certificates/* {path_to_occlum}/occlum_workspace/image/bin")

search_dirs = ["/home", "/hdd", "/etc", os.path.expanduser("~")]

inferONNX_path = None
for search_dir in search_dirs:
    for root, dirs, files in os.walk(search_dir):
        if "InferONNX" in dirs:
            inferONNX_path = os.path.join(root, "InferONNX")
            print("Found InferONNX directory at:", os.path.join(root, "InferONNX"))
            break
if not inferONNX_path:
    print("inferONNX path cannot be found!")
    exit(1)

path_to_scripts = inferONNX_path + "/scripts"
tag_file_path = inferONNX_path + '/src/tls_support/tag_file.txt'
if not os.path.exists(tag_file_path):
    with open(tag_file_path, "w") as f:
        f.write("")

no_ssl_tag_file_path = inferONNX_path + '/src/tls_support/tag_file.txt'
if not os.path.exists(no_ssl_tag_file_path):
    with open(no_ssl_tag_file_path, "w") as f:
        f.write("")

os.system(f"python3 {path_to_scripts}/run_tests_cpu.py tls_memory_only 1 {inferONNX_path} {path_to_occlum}")


#os.system(f"python3 {path_to_scripts}/run_tests_cpu.py on_disk entire 1 {inferONNX_path} {path_to_occlum}")
#os.system(f"python3 {path_to_scripts}/run_tests_occlum.py on_disk partitions 1 {inferONNX_path} {path_to_occlum}")
#os.system(f"python3 {path_to_scripts}/run_tests_occlum.py memory_only entire 1 {inferONNX_path} {path_to_occlum}")
#os.system(f"python3 {path_to_scripts}/run_tests_cpu.py on_disk 1 {inferONNX_path} {path_to_occlum}")
#os.system(f"python3 {path_to_scripts}/run_tests_cpu.py memory_only 1 {inferONNX_path} {path_to_occlum}")

