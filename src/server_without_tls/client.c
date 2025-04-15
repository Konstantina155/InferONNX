#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h> 
#include <unistd.h>
#include <assert.h>
#include <sys/time.h>
#include <errno.h>
#include <math.h>
#include <sys/time.h>
#include <stdbool.h>

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define SERVER_PORT 9997
#define SERVER_NAME "127.0.0.1"

#define BUF_SIZE 4096
#define TAG_SIZE 16

typedef struct __attribute__((packed)) {
    int command;
    int id;
    int num_models;
    int num_inputs;
    char **names;
    int *size_models;
    uint8_t **models;
    int* size_inputs;
    float **input;
    unsigned char **tags;
    int tokenizer_size;
    uint8_t *tokenizer;
} request;

/* HELPER FUNCTIONS */
int
size_of_file(FILE *fd)
{
    if (fseek(fd, 0, SEEK_END) != 0) {
        fprintf(stderr, "Error seeking end of file\n");
        return -1;
    }

    long size = ftell(fd);
    fprintf(stderr, "Size of file: %ld\n", size);
    if (size == -1) {
        fprintf(stderr, "Error getting size of file\n");
        return -1;
    }

    return (int)size;
}

static
size_t get_array_size(void **array)
{
    assert(array);

    size_t size = 0;
    while (array[size]) {
        size++;
    }
    return size;
}

static bool
is_hex(const char *str)
{
    assert(str);
    for (int i = 0; i < TAG_SIZE * 2; i++) {
        char c = str[i];
        if (!((c >= '0' && c <= '9') || 
              (c >= 'A' && c <= 'F') || 
              (c >= 'a' && c <= 'f'))) {
            return false;
        }
    }
    return true;
}

/* FILE OPERATIONS */
static size_t *
decode_pb(FILE *fd)
{
    uint8_t wire_type;
    uint64_t varint_value;
    static size_t shape[4] = {0, 0, 0, 0};

    int k = 0;
    uint8_t byte;
    while (fread(&byte, sizeof(uint8_t), 1, fd) == 1) {
        wire_type = byte & 0x07;
                
        if (wire_type == 0) { // Varint
            varint_value = 0;
            int shift = 0;
            do {
                if (fread(&byte, sizeof(uint8_t), 1, fd) != 1) break;
                varint_value |= ((uint64_t)(byte & 0x7F)) << (7 * shift);
                shift++;
            } while (byte & 0x80);
                        
            if (k < 4) 
                shape[k] = varint_value;
            
            k++;
        } else {
            break;
        }
    }

    if (k <= 4) 
        shape[k-1] = 0;

    fseek(fd, 0, SEEK_SET);

    return shape;
}

static float *
add_shape_to_array(float *image, size_t *shape)
{
    int size = (int)shape[0] * (int)shape[1] * (int)shape[2] * (int)shape[3];
    float *temp_image = (float *)realloc(NULL, (size + 4) * sizeof(float));
    if (!temp_image) {
        fprintf(stderr, "Error allocating memory for temporary image buffer to add the shape at the first 4 bytes\n");
        return NULL;
    }

    memcpy(temp_image + 4, image, size * sizeof(float));
    free(image);

    temp_image[0] = shape[0];
    temp_image[1] = shape[1];
    temp_image[2] = shape[2];
    temp_image[3] = shape[3];

    return temp_image;
}

static void *
assign_into_array(FILE *fd, int size, int element_size)
{
    void *data = malloc(size * element_size);
    if (!data) {
        fprintf(stderr, "Error allocating memory for file data\n");
        return NULL;
    }
    assert(fread(data, element_size, size, fd) == (size_t)size);
    return data;
}

FILE *
open_model_input(char *model_input)
{
    FILE *fd = fopen(model_input, "rb");
    if (!fd) {
        fprintf(stderr, "Error opening model_input file\n");
        return NULL;
    }

    return fd;
}

/* STRUCT OPERATIONS */
size_t
calculate_buffer_size(const request* req)
{
    size_t bytes = 0;

    // Int field -> command + id + num_models + num_inputs
    bytes += 4 * sizeof(int);

    int size = req->num_models;
    int num_inputs = req->num_inputs;
    fprintf(stderr, "num models: %d\n", size);

    // Char** field
    if (req->names != NULL && size > 0) {
        for (int i = 0; i < size; ++i) {
            bytes += strlen(req->names[i]) + 1;
        }
    }

    // Int* field
    if (req->size_models != NULL && size > 0) {
        bytes += size * sizeof(int);
    }

    // Uint8_t** field
    if (req->models != NULL && size > 0) {
        for (int i = 0; i < size; ++i) {
            bytes += req->size_models[i];
        }
    }

    // Int* field
    if (req->size_inputs != NULL && num_inputs > 0) {
        bytes += num_inputs * sizeof(int);
    }

    // Float** field
    if (req->input != NULL && num_inputs > 0) {
        for (int i = 0; i < req->num_inputs; ++i) {
            if (req->input[i] != NULL) {
                bytes += req->size_inputs[i] * sizeof(float);
            }
        }
    }

    // Unsigned char** field
    if (req->tags != NULL && size > 0) {
        for (int i = 0; i < size; ++i) {
            bytes += TAG_SIZE * 2;
        }
    }

    // Int field -> tokenizer_size
    bytes += sizeof(int);
    
    // Uint8_t* tokenizer field
    if (req->tokenizer != NULL && req->tokenizer_size > 0) {
        bytes += req->tokenizer_size * sizeof(uint8_t);
    }

    return bytes;
}

char * 
serialize_client_request(const request* req, ssize_t buffer_len)
{
    char *buffer = (char *)malloc((buffer_len + 1) * sizeof(char));
    assert(buffer);
    size_t bytes = 0;

    // Int field -> command + id + num_models + num_inputs
    memcpy(buffer + bytes, &req->command, sizeof(int));
    bytes += sizeof(int);
    memcpy(buffer + bytes, &req->id, sizeof(int));
    bytes += sizeof(int);
    memcpy(buffer + bytes, &req->num_models, sizeof(int));
    bytes += sizeof(int);
    memcpy(buffer + bytes, &req->num_inputs, sizeof(int));
    bytes += sizeof(int);

    int size = req->num_models;
    int num_inputs = req->num_inputs;

    // Char** field -> names
    if (req->names != NULL && size > 0) {
        for (int i = 0; i < size; ++i) {
            strcpy(buffer + bytes, req->names[i]);
            bytes += strlen(req->names[i]) + 1;
        }
    }

    // Int* field -> size_models
    if (req->size_models != NULL && size > 0) {
        memcpy(buffer + bytes, req->size_models, size * sizeof(int));
        bytes += size * sizeof(int);
    }

    // Uint8_t** field -> models
    if (req->models != NULL && size > 0) {
        for (int i = 0; i < size; ++i) {
            if (req->models[i] != NULL && req->size_models[i] > 0) {
                memcpy(buffer + bytes, req->models[i], req->size_models[i]);
                bytes += req->size_models[i];
            }
        }
    }

    // Int* field -> size_inputs
    if (req->size_inputs != NULL && num_inputs > 0) {
        memcpy(buffer + bytes, req->size_inputs, num_inputs * sizeof(int));
        bytes += num_inputs * sizeof(int);
    }

    // Float** field -> input
    if (req->input != NULL && req->num_inputs > 0) {
        for (int i = 0; i < req->num_inputs; ++i) {
            if (req->input[i] != NULL) {
                memcpy(buffer + bytes, req->input[i], req->size_inputs[i] * sizeof(float));
                bytes += req->size_inputs[i] * sizeof(float);
            }
        }
    }

    // Unsigned char** field -> tags
    if (req->tags != NULL && size > 0) {
        for (int i = 0; i < size; ++i) {
            memcpy(buffer + bytes, req->tags[i], TAG_SIZE * 2);
            bytes += TAG_SIZE * 2;
        }
    }

    // Int field -> tokenizer_size
    memcpy(buffer + bytes, &req->tokenizer_size, sizeof(int));
    bytes += sizeof(int);

    // Uint8_t* field -> tokenizer
    if (req->tokenizer != NULL && req->tokenizer_size > 0) {
        memcpy(buffer + bytes, req->tokenizer, req->tokenizer_size * sizeof(uint8_t));
        bytes += req->tokenizer_size * sizeof(uint8_t);
    }

    return buffer;
}

void
free_request(request *req_original)
{
    int size = req_original->num_models;
    int num_inputs = req_original->num_inputs;
    if (req_original->names != NULL) { 
        free(req_original->names);
    }
    if (req_original->size_models != NULL) {
        free(req_original->size_models);
    }
    if (req_original->models != NULL) {
        for (int i = 0; i < size; ++i) {
            free(req_original->models[i]);
        }
        free(req_original->models);
    }
    if (req_original->tags != NULL) {
        for (int i = 0; i < size; ++i) {
            free(req_original->tags[i]);
        }
        free(req_original->tags);
    }
    free(req_original->size_inputs);
    if (req_original->input != NULL) {
        for (int i = 0; i < num_inputs; ++i) {
            free(req_original->input[i]);
        }
        free(req_original->input);
    }
    if (req_original->tokenizer != NULL) {
        free(req_original->tokenizer);
    }
}

void
send_request(char *client_request, ssize_t request_len, int mode)
{
    struct timeval t1, t2;
    double elapsed_time;

    struct sockaddr_in addr;
    int client_fd, ret = 1;
    ssize_t bytes_read = 0;
    char input[BUF_SIZE];

    /*
     * 0. Opening a socket
     */
    fprintf(stderr, "Opening a socket...");
    client_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (client_fd < 0) {
        fprintf(stderr, " failed\n   socket returned %d\n", client_fd);
        return;
    }
    fprintf(stderr, " ok\n");
    
    addr.sin_family = AF_INET;
    addr.sin_port = htons(SERVER_PORT);
    addr.sin_addr.s_addr = inet_addr(SERVER_NAME);

    /*
     * 1. Start the connection
     */
    fprintf(stderr, "Connecting to %s/%d...", SERVER_NAME, SERVER_PORT);
    if ((ret = connect(client_fd, (struct sockaddr *)&addr, sizeof(addr))) < 0) {
        fprintf(stderr, " failed\n   connect returned %d\n", ret);
        return;
    }

    /*
     * 2. Write the GET request
     */
    int total_written = 0;

    fprintf(stderr, "\nWrite to server:");

    int request_size = (int)request_len;
    fprintf(stderr, "Request size: %ld, request_size (int): %d\n", request_len, request_size);
    
    gettimeofday(&t1, NULL);

    if ((total_written = send(client_fd, &request_size, sizeof(request_size), 0) < 0)) {
        fprintf(stderr, "Error sending request size\n");
        return;
    }
    fprintf(stderr, " %d bytes\n", total_written);

    if ((total_written = send(client_fd, client_request, request_size, 0)) < 0) {
        fprintf(stderr, "Error sending request\n");
        return;
    }

    fprintf(stderr, "Bytes written for message: %d\n\n", total_written);

    /*
     * 7. Read the response
     */
    fprintf(stderr, "Read from server:");

    if ((bytes_read = read(client_fd, input, sizeof(input) - 1)) < 0) {
        fprintf(stderr, "Connection closed by server\n");
        close(client_fd);
        return;
    }
    input[bytes_read] = '\0';

    gettimeofday(&t2, NULL);
    elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0;
    elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;

    if (mode == 1) {
        FILE *fd = NULL;
#ifdef USE_AES
        fd = fopen("../inference_time_cpu_memory_only_aes.txt", "a");
#else
    #ifdef USE_MEMORY_ONLY
        fd = fopen("../inference_time_cpu_memory_only_no_aes.txt", "a");
    #else
        fd = fopen("../inference_time_cpu_on_disk_no_aes.txt", "a");
    #endif
#endif

        if (!fd) {
            fprintf(stderr, "\nError opening inference_time_cpu_...!\n");
            return;
        }
        if (fprintf(fd, "Total time - client: %f ms\n", elapsed_time) < 0) {
            fprintf(stderr, "Error writing to file\n");
            return;
        }
        fclose(fd);
    }

    fprintf(stdout, " %ld bytes \nMessage from server: %s\n Connection was closed gracefully\n", bytes_read, input);

    close(client_fd);
}

int
filter_dir(const char *dir_path, const char *name)
{
    struct stat st;
    char full_path[1024];

    snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, name);

    if (stat(full_path, &st) != 0) {
        perror("stat");
        return 0;
    }

    return S_ISDIR(st.st_mode);
}

void
send_models(char **input_files, struct dirent **namelist, const char *dir_path, int num_models)
{
    assert(namelist);
    assert(input_files);
    assert(num_models > 0);

    request req_original;

    int temp_calculated_shape = 0;
    size_t *temp_shape = NULL;

    int num_inputs = get_array_size((void **)input_files) - 1;
    req_original.size_inputs = (int *)malloc(num_inputs * sizeof(int));
    assert(req_original.size_inputs);

    req_original.num_inputs = num_inputs;
    fprintf(stderr, "Number of inputs: %d\n", num_inputs);
    
    req_original.input = (float **)malloc(num_inputs * sizeof(float *));
    assert(req_original.input);
    for (int i = 0; i < num_inputs; i++) {
        FILE *fd = open_model_input(input_files[i]);
        if (!fd) {
            fprintf(stderr, "Error opening file %s\n", input_files[i]);
            return;
        }

        temp_shape = decode_pb(fd);
        temp_calculated_shape = (temp_shape[0] * temp_shape[1] * temp_shape[2] * temp_shape[3]) + 4;

        req_original.input[i] = (float *)assign_into_array(fd, temp_calculated_shape - 4, sizeof(float));
        if (!req_original.input) {
            fprintf(stderr, "Error assigning image to array\n");
            return;
        }

        req_original.input[i] = add_shape_to_array(req_original.input[i], temp_shape);
        if (!req_original.input[i]) {
            fprintf(stderr, "Error adding shape to array: %d\n", i);
            return;
        }

        req_original.size_inputs[i] = temp_calculated_shape;

        fclose(fd);
    }

    req_original.command = 0;
    req_original.id = -1;

    req_original.names = (char **) malloc((num_models + 1) * sizeof(char *));
    assert(req_original.names);

    req_original.num_models = 0;
    for (int i = 0; i < num_models; i++) {
        if (filter_dir(dir_path, namelist[i]->d_name) == 0) {
            req_original.names[req_original.num_models] = namelist[i]->d_name;
            fprintf(stderr, "Model: %s\n", req_original.names[req_original.num_models]);
            req_original.num_models++;
        }
    }

    fprintf(stderr, "Number of models: %d\n", req_original.num_models);
    req_original.names = realloc(req_original.names, req_original.num_models * sizeof(char *));
    if (!req_original.names) {
        perror("realloc");
        return;
    }

    req_original.models = (uint8_t **) malloc((req_original.num_models + 1) * sizeof(uint8_t *));
    assert(req_original.models);
    req_original.size_models = (int *) malloc(req_original.num_models * sizeof(int));
    assert(req_original.size_models);
    for (int i = 0; i < req_original.num_models; i++) {
        char full_path[1024];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, req_original.names[i]);

        FILE *fd = fopen(full_path, "rb");
        if (!fd) {
            fprintf(stderr, "Error opening file %s\n", full_path);
            free_request(&req_original);
            return;
        }

       if (fseek(fd, 0, SEEK_END) != 0) {
            fprintf(stderr, "Error seeking end of file\n");
            return;
        }

        long size = ftell(fd);
        fprintf(stderr, "Size of file: %ld\n", size);
        if (size == -1) {
            fprintf(stderr, "Error getting size of file: %s\n", full_path);
            free_request(&req_original);
            return;
        }

        if (fseek(fd, 0, SEEK_SET) != 0) {
            fprintf(stderr, "Error seeking beginning of file\n");
            return;
        }

        req_original.models[i] = (uint8_t *) malloc(size * sizeof(uint8_t));
        if (!req_original.models[i]) {
            fprintf(stderr, "Error allocating memory for file data\n");
            return;
        }

        if (fread(req_original.models[i], sizeof(uint8_t), size, fd) != (size_t)size) {
            fprintf(stderr, "Error reading file\n");
            free(req_original.models[i]);
            return;
        }
        req_original.size_models[i] = size;
        fclose(fd);
    }
    req_original.models[req_original.num_models] = NULL;

    req_original.tags = NULL;
    req_original.tokenizer_size = 0;
    req_original.tokenizer = NULL;

    size_t bufLen = calculate_buffer_size(&req_original);

    char *buffer = serialize_client_request(&req_original, bufLen);
    buffer[bufLen] = '\0';

    FILE *fd = NULL;
#ifdef USE_AES
    fd = fopen("../inference_time_cpu_memory_only_aes.txt", "a");
#else
    #ifdef USE_MEMORY_ONLY
        fd = fopen("../inference_time_cpu_memory_only_no_aes.txt", "a");
    #else
        fd = fopen("../inference_time_cpu_on_disk_no_aes.txt", "a");
    #endif
#endif

    if (!fd) {
        fprintf(stderr, "\nError opening inference_time_cpu_...!\n");
        return;
    }
    fprintf(fd, "\nModel: %s\n", req_original.names[0]);
    fclose(fd); 

    send_request(buffer, bufLen, 0);
    
    free_request(&req_original);
    free(buffer);
}

void
send_inputs(char **input_names, int id, unsigned char **tags, int size_tags)
{
    fprintf(stderr, "size_tags: %d\n", size_tags);
    request req_original;

    req_original.command = 1;
    req_original.id = id;
    req_original.num_models = size_tags;
    req_original.names = NULL;
    req_original.size_models = NULL;
    req_original.models = NULL;

    if (size_tags == 0) {
        req_original.tags = NULL;
    } else {
        req_original.tags = (unsigned char **) malloc((size_tags + 1) * sizeof(unsigned char *));
        for (int i = 0; i < size_tags; i++) {
            req_original.tags[i] = (unsigned char *)malloc(TAG_SIZE * 2 * sizeof(unsigned char));
            if (!req_original.tags[i]) {
                fprintf(stderr, "Error allocating memory for tags\n");
                return;
            }
            memcpy(req_original.tags[i], tags[i], TAG_SIZE * 2);
        }
        req_original.tags[size_tags] = NULL;
    }

    int temp_calculated_shape = 0;
    size_t *temp_shape = NULL;

    int num_inputs = get_array_size((void **)input_names);
    req_original.size_inputs = (int *)malloc(num_inputs * sizeof(int));
    assert(req_original.size_inputs);
  
    req_original.num_inputs = num_inputs;
    
    req_original.input = (float **)malloc(num_inputs * sizeof(float *));
    assert(req_original.input);
    fprintf(stderr, "Number of inputs: %d\n", num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        FILE *fd = open_model_input(input_names[i]);
        if (!fd) {
            fprintf(stderr, "Error opening file %s\n", input_names[i]);
            return;
        }

        fprintf(stderr, "Input: %s\n", input_names[i]);
        if (strstr(input_names[i], "tokenizer.json") != NULL) {
            int size = size_of_file(fd);
            if (size < 0) {
                return;
            }
            rewind(fd);
            
            free(req_original.input);
            free(req_original.size_inputs);
            req_original.num_inputs = 0;
            req_original.size_inputs = NULL;
            req_original.input = NULL;
            req_original.tokenizer_size = size;
            req_original.tokenizer = (uint8_t *)assign_into_array(fd, size, sizeof(uint8_t));
            if (!req_original.tokenizer) {
                fprintf(stderr, "Error assigning image to array\n");
                return;
            }
        } else {
            fprintf(stderr, "Not tokenizer\n");
            temp_shape = decode_pb(fd);
            temp_calculated_shape = (temp_shape[0] * temp_shape[1] * temp_shape[2] * temp_shape[3]) + 4;

            req_original.input[i] = (float *)assign_into_array(fd, temp_calculated_shape - 4, sizeof(float));
            if (!req_original.input) {
                fprintf(stderr, "Error assigning image to array\n");
                return;
            }

            req_original.input[i] = add_shape_to_array(req_original.input[i], temp_shape);
            if (!req_original.input[i]) {
                fprintf(stderr, "Error adding shape to array: %d\n", i);
                return;
            }

            req_original.size_inputs[i] = temp_calculated_shape;
            fprintf(stderr, "Size of input: %d\n", req_original.size_inputs[i]);

            fclose(fd);

            req_original.tokenizer_size = 0;
            req_original.tokenizer = NULL;
        }
    }
    
    size_t bufLen = calculate_buffer_size(&req_original);

    char *buffer = serialize_client_request(&req_original, bufLen);
    buffer[bufLen] = '\0';
    
    send_request(buffer, bufLen, 1);
    
    free_request(&req_original);
    free(buffer);
}

void
send_quit()
{
    request req_original;
    req_original.command = 2;
    req_original.id = -1;
    req_original.num_models = 0;
    req_original.num_inputs = 0;
    req_original.names = NULL;
    req_original.size_models = NULL;
    req_original.size_inputs = NULL;
    req_original.models = NULL;
    req_original.input = NULL;
    req_original.tags = NULL;
    req_original.tokenizer = NULL;
    req_original.tokenizer_size = 0;
    size_t bufLen = calculate_buffer_size(&req_original);
    char *buffer = serialize_client_request(&req_original, bufLen);
    fprintf(stderr, "%ld", bufLen);
    send_request(buffer, bufLen, 2);
    free_request(&req_original);
    free(buffer);
}

unsigned char **
get_tags(char *filename)
{
    FILE *fd = fopen(filename, "r");
    if (!fd) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return NULL;
    }
    
    int num_lines = 0;
    int ch;

    while ((ch = fgetc(fd)) != EOF) {
        if (ch == '\n') {
            num_lines++;
        }
    }   
    rewind(fd);

    unsigned char **data = (unsigned char **)malloc((num_lines + 2) * sizeof(unsigned char *));
    if (!data) {
        fprintf(stderr, "Memory allocation for %s failed\n", filename);
        return NULL;
    }

    int line_size = TAG_SIZE * 2;
    int i = 0;
    char *line = NULL;
    size_t len = 0;
    ssize_t read_data;
    while ((read_data = getline(&line, &len, fd)) != -1) {
        line[strcspn(line, "\r\n")] = 0;
        if ((int)strlen(line) != line_size) {
            continue;
        }
        if (is_hex(line) == false) {
            fprintf(stderr, "Invalid tag\n");
            free(line);
            for (int j = 0; j < i; j++) {
                free(data[j]);
            }
            free(data);
            fclose(fd);
            return NULL;
        }
        data[i] = (unsigned char *)strdup(line);
        data[i][line_size] = '\0';
        i++;
    }
    free(line);
    data[num_lines + 1] = NULL;

    fclose(fd);

    return data;
}

int
custom_strverscmp(const char *s1, const char *s2)
{
    char *p1 = (char *)s1;
    char *p2 = (char *)s2;

    while (*p1 && *p2) {
        if (isdigit(*p1) && isdigit(*p2)) {
            long val1 = strtol(p1, &p1, 10);
            long val2 = strtol(p2, &p2, 10);
            if (val1 != val2) {
                return val1 - val2;
            }
        } else {
            if (*p1 != *p2) {
                return *p1 - *p2;
            }
            p1++;
            p2++;
        }
    }
    return *p1 - *p2;
}

int
version_sort(const struct dirent **a, const struct dirent **b)
{
    return custom_strverscmp((*a)->d_name, (*b)->d_name);
}

int
main(int argc, char *argv[]) 
{
    if (argc < 2 || (strcmp(argv[1], "models") != 0 && strcmp(argv[1], "inputs") != 0 && strcmp(argv[1], "quit") != 0)) {
        fprintf(stderr, "Usage: %s 'inputs' <model_id> <tag_file> <model_input#1> ... <model_input#N> OR\n       %s 'models' <model_input#1> ... <model_input#N> <model_path> OR\n       %s 'quit'\n", argv[0], argv[0], argv[0]);
        return -1;
    }

    if (strcmp(argv[1], "models") == 0) {

        if (argc < 4) {
            fprintf(stderr, "Usage: %s 'models' <model_input> <directory>\n", argv[0]);
            return -1;
        }

        struct dirent **namelist;
        int num_models;

        const char *path = argv[argc - 1];
        num_models = scandir(path, &namelist, NULL, version_sort);

        if (num_models == -1) {
            perror("scandir");
            return 1;
        }

        send_models(argv + 2, namelist, path, num_models);

        for (int i = 0; i < num_models; i++) {
            free(namelist[i]);
        }
        free(namelist);
    } else if (strcmp(argv[1], "inputs") == 0) {

        int number = 3;
        unsigned char **tags = NULL;
        size_t num_tags = 0;

        char *endptr;
        long num_model = strtol(argv[2], &endptr, 10);
        if (*endptr != '\0') {
            fprintf(stderr, "Invalid model id\n");
            return -1;
        }
        int id = (int)num_model;

#if USE_AES == 1 && USE_MEMORY_ONLY == 0     
        if (argc < 5) {
            fprintf(stderr, "Usage: %s 'inputs' <model_id> <tag_file> <model_input#1> ... <model_input#N>\n", argv[0]);
            return -1;
        }

        number = 4;

        tags = get_tags(argv[3]);
        if (!tags) {
            fprintf(stderr, "File %s does not consist of tags\n", argv[3]);
            return -1;
        }

        num_tags = get_array_size((void **)tags);
        fprintf(stderr, "tag size: %ld\n", num_tags);
#else
        if (argc < 4) {
            fprintf(stderr, "Usage: %s 'inputs' <model_id> <model_input#1> ... <model_input#N>\n", argv[0]);
            return -1;
        }
#endif

        send_inputs(argv + number, id, tags, num_tags);
         
        if (tags) {
            for (int i = 0; tags[i]; i++) {
                free(tags[i]);
            }
            free(tags);
        }

    } else if (strcmp(argv[1], "quit") == 0) {
        send_quit();
    } else {
        fprintf(stderr, "Invalid command\n");
        return -1;
    }

    return 0;
}