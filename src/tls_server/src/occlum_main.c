#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <inference.h>
#include <ssl_crypto.h>

/* HELPER FUNCTIONS */
static void
free_array(void **array, int length)
{
    assert(array);
    for (int i = 0; i < length; ++i) {
        free(array[i]);
    }
    free(array);
}

static bool
contains_empty_name(char **names, int lenght)
{
    assert(names);
    for (int i = 0; i < lenght; ++i) {
        if (strlen(names[i]) == 0) return true; 
    }

    return false;
}

char **
add_path_to_names(char **names, int size_names)
{
    assert(names);

    const char *home_dir = "/bin";

    for (int i = 0; i < size_names; ++i) {
        char path[256];
        snprintf(path, sizeof(path), "%s/encrypted_models/", home_dir);
        strcat(path, names[i]);
        free(names[i]);
        names[i] = malloc((strlen(path) +1) * sizeof(char));
        if (!names[i]) {
            fprintf(stderr, "Memory allocation failed for adding path to names\n");
            free_array((void **) names, i);
            return NULL;
        }
        strcpy(names[i], path);
    }

    return names;
}

/* STRUCT OPERATIONS*/
encrypted_models_info *
initialize_encrypted_models_info(int num_models)
{
    encrypted_models_info *m = (encrypted_models_info *) malloc(sizeof(encrypted_models_info));
    if (!m) {
        fprintf(stderr, "Memory allocation failed for encrypted_models_info\n");
        return NULL;
    }

    memset(m->key, 0, KEY_BYTES);
    memset(m->IV, 0, IV_BYTES);
    memset(m->AAD, 0, ADD_DATA_BYTES);
    m->encrypted_model = (unsigned char **) malloc(num_models * sizeof(unsigned char *));
    m->tag = (unsigned char **) malloc(num_models * sizeof(unsigned char *));
    for (int i = 0; i < num_models; ++i) {
        m->tag[i] = (unsigned char *) malloc(TAG_BYTES * sizeof(unsigned char));
        assert(m->tag[i]);
        memset(m->tag[i], 0, TAG_BYTES);   
    }

    return m;
}

client_result *
initialize_client_result()
{
    client_result *c_l = (client_result *) malloc(sizeof(client_result));
    if (!c_l) {
        fprintf(stderr, "Memory allocation failed for client_result\n");
        return NULL;
    }

    c_l->result = NULL;
    c_l->tag = NULL;

    return c_l;
}

void
free_encrypted_models_info(encrypted_models_info *m, int num_models)
{
    assert(m);
    for (int i = 0; i < num_models; ++i) {
        free(m->encrypted_model[i]);
        free(m->tag[i]);
    }
    free(m->encrypted_model);
    free(m->tag);
    free(m);
}

void
free_client_result(client_result *c_l)
{
    assert(c_l);
    free(c_l->result);
    free(c_l);
}

char *
save_model(char *name, unsigned char *model, size_t size_model)
{
    assert(name);
    assert(model);

    FILE *fd;
    fd = fopen(name, "wb");
    if (!fd) {
        fprintf(stderr, "Error opening file %s\n", name);
        return NULL;
    }
    size_t byteswritten = fwrite(model, sizeof(unsigned char), size_model, fd);
    if (byteswritten != size_model) {
        fprintf(stderr, "fwrite failed\n");
        fclose(fd);
        return NULL;
    }
    fclose(fd);

    return "OK";
}

encrypted_models_info *
encrypt_models(char **names, int size_names, unsigned char **models, int *size_models)
{
    mbedtls_ctr_drbg_context ctr_drbg;
    mbedtls_entropy_context entropy;
    mbedtls_gcm_context gcm;

    unsigned char key[KEY_BYTES];
    unsigned char iv[IV_BYTES];
    unsigned char add_data[ADD_DATA_BYTES];
    unsigned char tag_encr[TAG_BYTES];
    size_t olen;
    int ret;

    // The personalization string should be unique to the application in order to add some
    // personalized starting randomness to the random sources.
    char *pers = "aes generate key";

    mbedtls_entropy_init(&entropy);
    mbedtls_ctr_drbg_init(&ctr_drbg);

    // Seed the random number generator
    ret = mbedtls_ctr_drbg_seed(&ctr_drbg, mbedtls_entropy_func, &entropy, (unsigned char *)pers, strlen(pers));
    if (ret != 0) {
        fprintf(stderr, "mbedtls_ctr_drbg_seed() failed - returned -0x%04x\n", -ret);
        goto exit_encr;
    }

    memset(key, 0, KEY_BYTES);
    memset(iv, 0, IV_BYTES);
    memset(add_data, 0, ADD_DATA_BYTES);
    memset(tag_encr, 0, TAG_BYTES);

    // Generate random bytes for the key (32 Bytes)
    ret = mbedtls_ctr_drbg_random(&ctr_drbg, key, KEY_BYTES);
    if (ret != 0) {
        fprintf(stderr, "mbedtls_ctr_drbg_random failed to extract key - returned -0x%04x\n", -ret);
        goto exit_encr;
    }

    // Generate random bytes for the IV (12 Bytes)
    ret = mbedtls_ctr_drbg_random(&ctr_drbg, iv, IV_BYTES);
    if (ret != 0) {
        fprintf(stderr, "mbedtls_ctr_drbg_random failed to extract IV - returned -0x%04x\n", -ret);
        goto exit_encr;
    }

    // Generate random bytes for the add_data (64 Bytes)
    ret = mbedtls_ctr_drbg_random(&ctr_drbg, add_data, ADD_DATA_BYTES);
    if (ret != 0) {
        fprintf(stderr, "mbedtls_ctr_drbg_random failed to extract add_data - returned -0x%04x\n", -ret);
        goto exit_encr;
    }

    encrypted_models_info *m = initialize_encrypted_models_info(size_names);
    if (!m) {
        fprintf(stderr, "Memory allocation failed for encrypted_models_info\n");
        goto exit_encr;
    }

    memcpy(m->key, key, KEY_BYTES);
    memcpy(m->IV, iv, IV_BYTES);
    memcpy(m->AAD, add_data, ADD_DATA_BYTES);

    unsigned char *encrypted_model = NULL;
    size_t size_model = 0;
    for (int i = 0; i < size_names; ++i) {
        size_model = (size_t)size_models[i];
        encrypted_model = (unsigned char *)malloc(size_model);
        if (!encrypted_model) {
            ret = 1;
            fprintf(stderr, "Memory allocation for encrypted model `%s` failed\n", names[i]);
            goto exit_encr;
        }

        mbedtls_gcm_init(&gcm);

        // Initialize the GCM context with our key and desired cipher
        ret = mbedtls_gcm_setkey(&gcm,                      // GCM context to be initialized
                                MBEDTLS_CIPHER_ID_AES,     // cipher to use (a 128-bit block cipher)
                                key,                       // encryption key
                                KEY_BITS);                 // key bits
        if (ret != 0) {
            fprintf(stderr, "mbedtls_gcm_setkey failed to set the key for AES cipher in encryption process - returned -0x%04x\n", -ret);
            goto exit_encr;
        }

        // Start the GCM encryption process
        ret = mbedtls_gcm_starts(&gcm,                 // GCM context
                                MBEDTLS_GCM_ENCRYPT,   // mode
                                iv,                    // initialization vector
                                IV_BYTES);             // length of IV
        if (ret != 0) {
            fprintf(stderr, "mbedtls_gcm_starts failed to start the encryption process - returned -0x%04x\n", -ret);
            goto exit_encr;
        }

        // Set additional authenticated data (AAD)
        ret = mbedtls_gcm_update_ad(&gcm,              // GCM context
                                    add_data,          // additional data
                                    ADD_DATA_BYTES);   // length of AAD
        if (ret != 0) {
            fprintf(stderr, "mbedtls_gcm_starts failed to set the AAD in the encryption process - returned -0x%04x\n", -ret);
            goto exit_encr;
        }

        if (size_model > 32) {
            size_t rest_len = size_model - 32;

            // Encrypt the first 32 bytes
            ret = mbedtls_gcm_update(&gcm,               // GCM context
                                    models[i],           // input data
                                    32,                  // length of first 32 bytes of input data
                                    encrypted_model,     // output of encryption process for the first 32 bytes
                                    size_model,          // length of input data
                                    &olen);              // length of output data (expected 32)
            if (ret != 0) {
                fprintf(stderr, "mbedtls_gcm_update failed to encrypt the first 32 bytes of input data - returned -0x%04x\n", -ret);
                goto exit_encr;
            }
            if (olen != 32) {
                fprintf(stderr, "mbedtls_gcm_update failed to calculate olen in encryption process, expected 32 - returned -0x%04x\n", -ret);
                goto exit_encr;
            }

            // Encrypt the rest of the data
            ret = mbedtls_gcm_update(&gcm,                     // GCM context
                                    models[i] + 32,            // input data for the rest data
                                    rest_len,                  // length of the rest data
                                    encrypted_model + 32,      // output of encryption process for the rest data
                                    size_model - 32,           // length of the rest data
                                    &olen);                    // length of output data (expected rest_len)
            if (ret != 0) {
                fprintf(stderr, "mbedtls_gcm_update failed to encrypt the rest of the data - returned -0x%04x\n", -ret);
                goto exit_encr;
            }
            if (olen != rest_len) {
                fprintf(stderr, "mbedtls_gcm_update failed to calculate olen in encryption process, expected %ld - returned -0x%04x\n", rest_len, -ret);
                goto exit_encr;
            }
        } else {
            // Encrypt the whole model
            ret = mbedtls_gcm_update(&gcm,                // GCM context
                                    models[i],            // input data
                                    size_model,           // length of input data
                                    encrypted_model,      // output of encryption process
                                    size_model,           // length of input data
                                    &olen);               // length of output data (expected size_model)
            if (ret != 0) {
                fprintf(stderr, "mbedtls_gcm_update failed to encrypt the whole data - returned -0x%04x\n", -ret);
                goto exit_encr;
            }
            if (olen != size_model) {
                fprintf(stderr, "mbedtls_gcm_update failed to calculate olen in the encryption process, expected %ld - returned -0x%04x\n", size_model, -ret);
                goto exit_encr;
            }
        }

        // Finish the GCM encryption process and generate the tag in encryption process
        ret = mbedtls_gcm_finish(&gcm,           // GCM context
                                NULL,            // input data, here NULL
                                0,               // length of input data, here 0
                                &olen,           // length of output data, here olen
                                tag_encr,        // buffer for holding the tag
                                TAG_BYTES);      // length of the tag
        if (ret != 0) {
            fprintf(stderr, "mbedtls_gcm_finish failed to finish the encryption process and generate the tag - returned -0x%04x\n", -ret);
            goto exit_encr;
        }

        m->encrypted_model[i] = (unsigned char *)malloc(size_model * sizeof(unsigned char));
        if (!m->encrypted_model[i] || !m->tag[i]) {
            fprintf(stderr, "Memory allocation for m->encrypted_model || m->tag[i] failed\n");
            goto exit_encr;
        }
        memcpy(m->encrypted_model[i], encrypted_model, size_model);
        memcpy(m->tag[i], tag_encr, TAG_BYTES);

        if (!save_model(names[i], encrypted_model, size_model)) {
            fprintf(stderr, "Error saving model %s\n", names[i]);
            goto exit_encr;
        }
        free(encrypted_model);

        //print tag_encr in hex
        fprintf(stderr, "Tag in tag_encr: ");
        for (int j = 0; j < TAG_BYTES; ++j) {
            fprintf(stderr, "%02x", tag_encr[j]);
        }
        fprintf(stderr, "\n");

        fprintf(stderr, " Tag in m->tag: ");
        for (int j = 0; j < TAG_BYTES; ++j) {
            fprintf(stderr, "%02x", m->tag[i][j]);
        }
        fprintf(stderr, "\n");

        mbedtls_gcm_free(&gcm);
    }

exit_encr:
    mbedtls_ctr_drbg_free(&ctr_drbg);
    mbedtls_entropy_free(&entropy);

    if (ret != 0) {
        mbedtls_gcm_free(&gcm);
        fprintf(stderr, "FAILURE when encrypting model\n");
        return NULL;
    }

    return m;
}

void 
deserialize_client_request(const char* buf, request* req)
{
    size_t offset = 0;

    // int field -> command + id + num_models + num_inputs
    memcpy(&req->command, buf + offset, sizeof(int));
    offset += sizeof(int);
    memcpy(&req->id, buf + offset, sizeof(int));
    offset += sizeof(int);
    memcpy(&req->num_models, buf + offset, sizeof(int));
    offset += sizeof(int);
    memcpy(&req->num_inputs, buf + offset, sizeof(int));
    offset += sizeof(int);

    fprintf(stderr, "command: %d, id: %d, num_models: %d, num_inputs: %d\n", req->command, req->id, req->num_models, req->num_inputs);

    int size = req->num_models;
    int num_inputs = req->num_inputs;
    fprintf(stderr, "size: %d\n", size);

    if (req->command == 0) {
        // Char ** field -> names
        if (size > 0) {
            req->names = malloc((size + 1) * sizeof(char *));
            for (int i = 0; i < size; ++i) {
                req->names[i] = strdup(buf + offset);
                offset += strlen(buf + offset) + 1;
            }
            req->names[size] = NULL;
        } else {
            req->names = NULL;
        }

        // Int* field -> size_models
        if (size > 0) {
            req->size_models = malloc(size * sizeof(int));
            memcpy(req->size_models, buf + offset, size * sizeof(int));
            offset += size * sizeof(int);
        } else {
            req->size_models = NULL;
        }

        // Uint8_t** field -> models
        if (size > 0) {
            req->models = malloc(size * sizeof(uint8_t *));
            for (int i = 0; i < size; ++i) {
                req->models[i] = malloc(req->size_models[i]);
                memcpy(req->models[i], buf + offset, req->size_models[i]);
                offset += req->size_models[i];
            }
        } else {
            req->models = NULL;
        }

        // Int* field -> size_inputs
        if (num_inputs > 0) {
            req->size_inputs = malloc(num_inputs * sizeof(int));
            memcpy(req->size_inputs, buf + offset, num_inputs * sizeof(int));
            offset += num_inputs * sizeof(int);
        } else {
            req->size_inputs = NULL;
        }

        // Float ** field -> input
        if (num_inputs > 0) {
            fprintf(stderr, "num_inputs: %d\n", num_inputs);
            req->input = malloc(num_inputs * sizeof(float *));
            for (int i = 0; i < num_inputs; ++i) {
                fprintf(stderr, "input_size[%d]: %d\n", i, req->size_inputs[i]);
                req->input[i] = malloc(req->size_inputs[i] * sizeof(float));
                memcpy(req->input[i], buf + offset, req->size_inputs[i] * sizeof(float));
                offset += req->size_inputs[i] * sizeof(float);
            }
        } else {
            req->input = NULL;
        }
        req->tags = NULL;

        // Int field -> tokenizer_size
        memcpy(&req->tokenizer_size, buf + offset, sizeof(int));
        offset += sizeof(int);
        req->tokenizer = NULL;
    } else {
        // Int* field -> size_inputs
        if (num_inputs > 0) {
            req->size_inputs = malloc(num_inputs * sizeof(int));
            memcpy(req->size_inputs, buf + offset, num_inputs * sizeof(int));
            offset += num_inputs * sizeof(int);
        } else {
            req->size_inputs = NULL;
        }

        // Float ** field -> input
        if (num_inputs > 0) {
            fprintf(stderr, "num_inputs: %d\n", num_inputs);
            req->input = malloc(num_inputs * sizeof(float *));
            for (int i = 0; i < num_inputs; ++i) {
                fprintf(stderr, "input_size[%d]: %d\n", i, req->size_inputs[i]);
                req->input[i] = malloc(req->size_inputs[i] * sizeof(float));
                memcpy(req->input[i], buf + offset, req->size_inputs[i] * sizeof(float));
                offset += req->size_inputs[i] * sizeof(float);
            }
        } else {
            req->input = NULL;
        }

        // Unsigned char** field -> tags
        if (size > 0) {
            req->tags = malloc((size + 1) * sizeof(unsigned char *));
            for (int i = 0; i < size; ++i) {
                req->tags[i] = malloc(TAG_BYTES * 2);
                memcpy(req->tags[i], buf + offset, TAG_BYTES * 2);
                offset += TAG_BYTES * 2;
            }
            req->tags[size] = NULL;
        } else {
            req->tags = NULL;
        }

        req->names = NULL;
        req->size_models = NULL;
        req->models = NULL;
        req->num_models = 0;

        memcpy(&req->tokenizer_size, buf + offset, sizeof(int));
        offset += sizeof(int);

        // Uint8_t* field -> tokenizer
        if (req->tokenizer_size > 0) {
            req->tokenizer = malloc(req->tokenizer_size * sizeof(uint8_t));
            memcpy(req->tokenizer, buf + offset, req->tokenizer_size * sizeof(uint8_t));
            offset += req->tokenizer_size * sizeof(uint8_t);
        } else {
            req->tokenizer = NULL;
        }
    }
}

void
free_request(request *req_original)
{
    int size = req_original->num_models;
    int num_inputs = req_original->num_inputs;
    if (req_original->names != NULL) { 
        for (int i = 0; i < size; ++i) {
            free(req_original->names[i]);
        }
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
        for (int i = 0; req_original->tags[i] != NULL; ++i) {
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

client_result *
handle_request(char *client_request, onnx_table *table)
{
    assert(client_request);
    
    request req_copy;
    deserialize_client_request(client_request, &req_copy);

    int command = req_copy.command, id = req_copy.id;
    char **names = req_copy.names;
    uint8_t **models = req_copy.models;
    float **input = req_copy.input;
    int *size_inputs = req_copy.size_inputs;
    int num_inputs = req_copy.num_inputs;
    int num_models = req_copy.num_models;
    int *size_models = req_copy.size_models;
    unsigned char **tags = req_copy.tags;
    int tokenizer_size = req_copy.tokenizer_size;
    uint8_t* tokenizer = req_copy.tokenizer;

    int size = 0;
    client_result *c_l = initialize_client_result();
    switch (command) {
    case 0: {
        if (!names || num_models == 0 || !models || contains_empty_name(names, num_models) || (id != -1 || !input || tags || tokenizer_size != 0)) {
            fprintf(stderr, "Invalid request for MODEL\n");
            free_request(&req_copy);
            free(client_request);
            return NULL;
        }

        fprintf(stderr, "MODEL SIZE: %d\n", size_models[0]);

        names = add_path_to_names(names, num_models);

        if (find_duplicate_names_from_id(table, names)) {
            char *error = (char *) malloc(512 * sizeof(char));
            if (!error) {
                fprintf(stderr, "Error allocating memory for error\n");
                free_request(&req_copy);
                free(client_request);
                return NULL;
            }
            snprintf(error, 512, "Model is already in the onnx table");
            error[511] = '\0';
            c_l->size = 512;
            c_l->result = (unsigned char *) malloc((c_l->size + 1) * sizeof(unsigned char));
            if (!c_l->result) {
                fprintf(stderr, "Memory allocation failed for c_l->result in MODEL\n");
                free_request(&req_copy);
                free(client_request);
                return NULL;
            }
            memcpy(c_l->result, error, c_l->size);
            free_request(&req_copy);
            free(client_request);
            free(error);
            return c_l;
        }
        
        encrypted_models_info *me = encrypt_models(names, num_models, models, size_models);
        if (!me) {
            free_request(&req_copy);
            free(client_request);
            return NULL;
        }

        model *m = (model *) malloc(sizeof(model));
        assert(m);
        m->next = NULL;
        m->size = size;
        m->names = names;
        memcpy(m->key, me->key, KEY_BYTES);
        memcpy(m->IV, me->IV, IV_BYTES);
        memcpy(m->AAD, me->AAD, ADD_DATA_BYTES);
        m->inference_models = NULL;
        m->head = NULL;

        unsigned char **tags = (unsigned char **) malloc(num_models * sizeof(unsigned char *));
        if (!tags) {
            fprintf(stderr, "Memory allocation failed for tags in MODEL\n");
            free_request(&req_copy);
            free(client_request);
            free_encrypted_models_info(me, num_models);
            free(c_l->result);
            return NULL;
        }
        for (int i = 0; i < num_models; ++i) {
            tags[i] = (unsigned char *) malloc((TAG_BYTES * 2 + 1) * sizeof(unsigned char));
            if (!tags[i]) {
                fprintf(stderr, "Memory allocation failed for tags[i] in MODEL\n");
                free_request(&req_copy);
                free(client_request);
                free_encrypted_models_info(me, num_models);
                free(c_l->result);
                return NULL;
            }
            memset(tags[i], 0, TAG_BYTES * 2 + 1);
            for (int j = 0; j < TAG_BYTES; ++j) {
                sprintf((char *)(tags[i] + j * 2), "%02x", me->tag[i][j]);
            }
            tags[i][TAG_BYTES * 2] = '\0';
        }
  
        load_model_to_memory(&m, tags, num_models);
        
        for (int i = 0; i < num_models; ++i) {
            free(tags[i]);
        }
        free(tags);

    #if USE_MEMORY_ONLY == 0    
        c_l->tag = (unsigned char **) malloc((num_models + 1)* sizeof(unsigned char *));
        for (int i = 0; i < num_models; ++i) {
            c_l->tag[i] = (unsigned char *) malloc(TAG_BYTES * sizeof(unsigned char));
            if (!c_l->tag[i]) {
                fprintf(stderr, "Memory allocation failed for c_l->tag in MODEL\n");
                free_request(&req_copy);
                free(client_request);
                free_encrypted_models_info(me, num_models);
                free(c_l->result);
                return NULL;
            }
            memset(c_l->tag[i], 0, TAG_BYTES);
            memcpy(c_l->tag[i], me->tag[i], TAG_BYTES);
            
        }
        c_l->tag[num_models] = NULL;
    #endif

        char *id_str = insert_into_table(table, m);
        if (!id_str) {
            free_encrypted_models_info(me, num_models);
            free_request(&req_copy);
            free(client_request);
            return NULL;
        }
        
        free_request(&req_copy);
        print_table(table);

        size = strlen(id_str);
        c_l->result = (unsigned char *) malloc((size + 1) * sizeof(unsigned char));
        if (!c_l->result) {
            fprintf(stderr, "Memory allocation failed for c_l->result in MODEL\n");
            free_request(&req_copy);
            free(client_request);
            return NULL;
        }

        memcpy(c_l->result, id_str, size);
        free_encrypted_models_info(me, num_models);
        c_l->size = size;

        break;
    }   
    case 1: {
        for (int i = 0; i < num_inputs; ++i) {
            if ((size_inputs[i] == 0 || !input[i]) && tokenizer == NULL) {
                fprintf(stderr, "Invalid request for MODEL_INPUT for tokenizer\n");
                free_request(&req_copy);
                return NULL;
            }
        }

        if (names || id == -1 || num_models != 0) {
            fprintf(stderr, "Invalid request for MODEL_INPUT for rest fields\n");
            free_request(&req_copy);
            return NULL;
        }

 #if USE_MEMORY_ONLY == 1
        if (tags) {
            fprintf(stderr, "Invalid request for MODEL_INPUT for tags for cache\n");
            free_request(&req_copy);
            return NULL;
        }
#else
        if (!tags) {
            fprintf(stderr, "Invalid request for MODEL_INPUT for tags\n");
            free_request(&req_copy);
            return NULL;
        }
#endif

        for (int i = 0; i < num_inputs; i++) {
            fprintf(stderr, "INPUT SIZE[%d]: %d\n", i, size_inputs[i]);
        }

        int required_size = snprintf(NULL, 0, "%d", id);
        char id_str[required_size + 1];
        snprintf(id_str, required_size + 1, "%d", id);

        char *result = NULL;
        model *m = get_model(table, id_str);
        if (!m) {
            char *error = (char *) malloc(512 * sizeof(char));
            if (!error) {
                fprintf(stderr, "Error allocating memory for error\n");
                return NULL;
            }
            snprintf(error, 512, "Model with id %d not found\n", id);
            error[511] = '\0';
            c_l->size = 512;
            c_l->result = (unsigned char *) malloc((c_l->size + 1) * sizeof(unsigned char));
            if (!c_l->result) {
                fprintf(stderr, "Memory allocation failed for c_l->result in MODEL\n");
                free_request(&req_copy);
                free(client_request);
                return NULL;
            }
            memcpy(c_l->result, error, c_l->size);
            free_request(&req_copy);
            free(client_request);
            free(error);
            return c_l;
        }

#if USE_MEMORY_ONLY == 0
        result = inference_aes(input, num_inputs, tokenizer, tokenizer_size, m, tags, m->size);
#else
        result = inference_cache(input, num_inputs, m);
#endif

        if (!result) {
            free_request(&req_copy);
            free(client_request);
            return NULL;
        }

        free_request(&req_copy);

        int size = strlen(result);
        c_l->result = (unsigned char *) malloc((size + 1) * sizeof(unsigned char));
        if (!c_l->result) {
            fprintf(stderr, "Memory allocation failed for c_l->result in MODEL\n");
            free_request(&req_copy);
            free(client_request);
            return NULL;
        }
        memcpy(c_l->result, result, size);
        c_l->size = size;

        free(result);

        break;
    }
    default:
        fprintf(stderr, "Invalid command\n");
        free_request(&req_copy);
        free(client_request);
        return NULL;
    }

    free(client_request);
    return c_l;
}

// Debug for SSL
void
debug_ssl(void *ctx, int level,
                     const char *file, int line,
                     const char *str)
{
    ((void) level);

    fprintf((FILE *) ctx, "%s:%04d: %s", file, line, str);
    fflush((FILE *) ctx);
}

int
main(void)
{
#ifdef USE_SYS_TIME
    struct timeval t1, t2, t1_read, t2_read, t1_write, t2_write, t1_handshake, t2_handshake, t1_rest, t2_rest;
#endif
    double elapsed_time, elapsed_time_read, elapsed_time_write, elapsed_time_handshake, elapsed_time_rest;

    int ret;
    mbedtls_net_context listen_fd, client_fd;
    const char *pers = "ssl_server";

    mbedtls_entropy_context entropy;
    mbedtls_ctr_drbg_context ctr_drbg;
    mbedtls_ssl_context ssl;
    mbedtls_ssl_config conf;
    mbedtls_x509_crt srvcert;
    mbedtls_pk_context pkey;
#if defined(MBEDTLS_SSL_CACHE_C)
    mbedtls_ssl_cache_context cache;
#endif

    mbedtls_net_init(&listen_fd);
    mbedtls_net_init(&client_fd);
    mbedtls_ssl_init(&ssl);
    mbedtls_ssl_config_init(&conf);
#if defined(MBEDTLS_SSL_CACHE_C)
    mbedtls_ssl_cache_init(&cache);
#endif
    mbedtls_x509_crt_init(&srvcert);
    mbedtls_pk_init(&pkey);
    mbedtls_entropy_init(&entropy);
    mbedtls_ctr_drbg_init(&ctr_drbg);

#if defined(MBEDTLS_USE_PSA_CRYPTO)
    psa_status_t status = psa_crypto_init();
    if (status != PSA_SUCCESS) {
        fprintf(stderr, stderr, "Failed to initialize PSA Crypto implementation: %d\n",
                        (int) status);
        ret = MBEDTLS_ERR_SSL_HW_ACCEL_FAILED;
        goto exit;
    }
#endif /* MBEDTLS_USE_PSA_CRYPTO */

#if defined(MBEDTLS_DEBUG_C)
    mbedtls_debug_set_threshold(DEBUG_LEVEL);
#endif

    /*
     * 1. Seed the RNG
     */
    fprintf(stderr, "Seeding the random number generator...");
    fflush(stdout);

    if ((ret = mbedtls_ctr_drbg_seed(&ctr_drbg, mbedtls_entropy_func, &entropy,
                                     (const unsigned char *) pers,
                                     strlen(pers))) != 0) {
        fprintf(stderr, " failed\n   mbedtls_ctr_drbg_seed returned %d\n", ret);
        goto exit;
    }

    fprintf(stderr, " ok\n");

    /*
     * 2. Load the certificates and private RSA key
     */
    fprintf(stderr, "Loading the server cert and key...");
    fflush(stdout);

    /*
     * This demonstration program uses embedded test certificates.
     * Instead, you may want to use mbedtls_x509_crt_parse_file() to read the
     * server and CA certificates, as well as mbedtls_pk_parse_keyfile().
     */
    ret = mbedtls_x509_crt_parse_file(&srvcert, "/bin/cert.pem");
    if (ret != 0) {
        fprintf(stderr, " failed\n   mbedtls_x509_crt_parse_file returned %d\n", ret);
        goto exit;
    }

    ret =  mbedtls_pk_parse_keyfile(&pkey, "/bin/key.pem", NULL, mbedtls_ctr_drbg_random, &ctr_drbg);

    if (ret != 0) {
        fprintf(stderr, " failed\n   mbedtls_pk_parse_keyfile returned %d\n", ret);
        goto exit;
    }

    fprintf(stderr, " ok\n");

    /*
     * 3. Setup the listening TCP socket
     */
    fprintf(stderr, "Bind on https://localhost:9998/ ...");
    fflush(stdout);

    if ((ret = mbedtls_net_bind(&listen_fd, NULL, "9998", MBEDTLS_NET_PROTO_TCP)) != 0) {
        fprintf(stderr, " failed\n   mbedtls_net_bind returned %d\n", ret);
        goto exit;
    }

    fprintf(stderr, " ok\n");

    /*
     * 4. Setup stuff
     */
    fprintf(stderr, "Setting up the SSL data....");
    fflush(stdout);

    if ((ret = mbedtls_ssl_config_defaults(&conf,
                                           MBEDTLS_SSL_IS_SERVER,
                                           MBEDTLS_SSL_TRANSPORT_STREAM,
                                           MBEDTLS_SSL_PRESET_DEFAULT)) != 0) {
        fprintf(stderr, " failed\n   mbedtls_ssl_config_defaults returned %d\n", ret);
        goto exit;
    }

    mbedtls_ssl_conf_rng(&conf, mbedtls_ctr_drbg_random, &ctr_drbg);
    mbedtls_ssl_conf_dbg(&conf, debug_ssl, stdout);

#if defined(MBEDTLS_SSL_CACHE_C)
    mbedtls_ssl_conf_session_cache(&conf, &cache,
                                   mbedtls_ssl_cache_get,
                                   mbedtls_ssl_cache_set);
#endif

    mbedtls_ssl_conf_ca_chain(&conf, srvcert.next, NULL);
    if ((ret = mbedtls_ssl_conf_own_cert(&conf, &srvcert, &pkey)) != 0) {
        fprintf(stderr, " failed\n   mbedtls_ssl_conf_own_cert returned %d\n", ret);
        goto exit;
    }

    if ((ret = mbedtls_ssl_setup(&ssl, &conf)) != 0) {
        fprintf(stderr, " failed\n   mbedtls_ssl_setup returned %d\n", ret);
        goto exit;
    }

    fprintf(stderr, " ok\n");

    onnx_table *table = init_onnx_table(CAPACITY);
    char response[BUF_SIZE];
    long request_size = 0, response_size = 0;
    char *client_request = NULL;
    size_t bytes_read = 0;
    unsigned char buf[BUF_SIZE];
    char *endptr;

reset:
#ifdef MBEDTLS_ERROR_C
    if (ret != 0) {
        char error_buf[100];
        mbedtls_strerror(ret, error_buf, 100);
        fprintf(stderr, "Last error was: %d - %s\n", ret, error_buf);
    }
#endif

    mbedtls_net_free(&client_fd);

    mbedtls_ssl_session_reset(&ssl);

    /*
     * 3. Wait until a client connects
     */
    fprintf(stderr, "\n\nWaiting for a remote connection ...");
    fflush(stdout);

    if ((ret = mbedtls_net_accept(&listen_fd, &client_fd,
                                  NULL, 0, NULL)) != 0) {
        fprintf(stderr, " failed\n   mbedtls_net_accept returned %d\n", ret);
        goto exit;
    }

    mbedtls_ssl_set_bio(&ssl, &client_fd, mbedtls_net_send, mbedtls_net_recv, NULL);

    fprintf(stderr, " ok\n");

    /*
     * 5. Handshake
     */
#ifdef USE_SYS_TIME
    gettimeofday(&t1, NULL);
    gettimeofday(&t1_handshake, NULL);
#endif
    fprintf(stderr, "Performing the SSL/TLS handshake...");
    fflush(stdout);

    while ((ret = mbedtls_ssl_handshake(&ssl)) != 0) {
        if (ret != MBEDTLS_ERR_SSL_WANT_READ && ret != MBEDTLS_ERR_SSL_WANT_WRITE) {
            fprintf(stderr, " failed\n   mbedtls_ssl_handshake returned %d\n", ret);
            goto reset;
        }
    }
    fprintf(stderr, " ok\n\n");
#ifdef USE_SYS_TIME
    gettimeofday(&t2_handshake, NULL);
    gettimeofday(&t1_read, NULL);
#endif

    /*
     * 6. Read the Request
     */
    fprintf(stderr, "Read from client:");
    fflush(stdout);

    do {
        ret = mbedtls_ssl_read(&ssl, (unsigned char *) buf, BUF_SIZE);

        if (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE) {
            continue;
        }

        if (ret <= 0) {
            switch (ret) {
                case MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY:
                    fprintf(stderr, " Connection was closed gracefully\n\n");
                    break;

                case MBEDTLS_ERR_NET_CONN_RESET:
                    fprintf(stderr, " Connection was reset by peer\n");
                    break;

                default:
                    fprintf(stderr, " mbedtls_ssl_read returned -0x%x\n\n", ret);
                    goto reset;
            }

            break;
        }
        request_size = strtol((char *) buf, &endptr, 10);
        if (*endptr != '\0') {
            fprintf(stderr, "The size of the client_request is not an integer!\n");
            goto reset;
        }
        fprintf(stderr, "Bytes received: %d\n Message from client: %ld\n", ret, request_size);

        if (ret == 3 && request_size == 20) {
            fprintf(stderr, "Client wants to close the connection...\n");
            free_onnx_table(table);
            ret = 0;
            goto exit;
        }

        client_request = (char *) malloc((request_size +1) * sizeof(char));
        if (!client_request) {
            perror("Memory allocation failed for client_request");
            goto exit;
        }

        bytes_read = 0;
        while (bytes_read < (size_t) request_size) {
            ret = mbedtls_ssl_read(&ssl, (unsigned char *) (client_request + bytes_read), request_size - bytes_read);

            if (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE) {
                continue;
            }

            if (ret <= 0) {
                switch (ret) {
                    case MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY:
                        fprintf(stderr, " Connection was closed gracefully\n");
                        break;

                    case MBEDTLS_ERR_NET_CONN_RESET:
                        fprintf(stderr, " Connection was reset by peer\n");
                        break;

                    default:
                        fprintf(stderr, " mbedtls_ssl_read returned -0x%x\n", (unsigned int) -ret);
                        free(client_request);
                        goto reset;
                }

                break;
            }
            bytes_read += ret;
        }
        client_request[request_size] = '\0';

        fprintf(stderr, "Bytes received: %ld\n", bytes_read);

        if (ret > 0) {
            break;
        }
    } while (1);

#ifdef USE_SYS_TIME
    gettimeofday(&t2_read, NULL);
    gettimeofday(&t1_rest, NULL);
#endif

    client_result *c_l = handle_request(client_request, table);
    if (!c_l) {
        strcpy(response, "Invalid client_request from handle_request\n");
    } else {
        memcpy(response, c_l->result, c_l->size);
        int current_position = c_l->size;
        if (c_l->tag) {
            for (size_t i = 0; c_l->tag[i] != NULL; ++i) {
                response[current_position++] = ' '; // Add a space separator
                fprintf(stderr, "Tag: ");
                for (size_t j = 0; j < TAG_BYTES; ++j) {
                    fprintf(stderr, "%02x", c_l->tag[i][j]);
                    sprintf(response + current_position, "%02x", c_l->tag[i][j]);
                    current_position += 2;
                }
                fprintf(stderr, "\n"); 
                free(c_l->tag[i]);
            }
            free(c_l->tag);
        }
        response[current_position] = '\0';
        free_client_result(c_l);
    }

    /*
     * 7. Write the Response
     */
    fprintf(stderr, "Write to client:");
    fflush(stdout);

    fprintf(stderr, "\nSSL ciphersuite: %s\n", mbedtls_ssl_get_ciphersuite(&ssl));

    response_size = strlen(response);

#ifdef USE_SYS_TIME
    gettimeofday(&t2_rest, NULL);
    gettimeofday(&t1_write, NULL);
#endif

    while ((ret = mbedtls_ssl_write(&ssl, (unsigned char *) response, response_size)) <= 0) {
        if (ret == MBEDTLS_ERR_NET_CONN_RESET) {
            fprintf(stderr, " failed\n   peer closed the connection\n");
            goto reset;
        }

        if (ret != MBEDTLS_ERR_SSL_WANT_READ && ret != MBEDTLS_ERR_SSL_WANT_WRITE) {
            fprintf(stderr, " failed\n   mbedtls_ssl_write returned %d\n", ret);
            goto exit;
        }
    }

#ifdef USE_SYS_TIME
    gettimeofday(&t2_write, NULL);

    gettimeofday(&t2, NULL);
    elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

    elapsed_time_handshake = (t2_handshake.tv_sec - t1_handshake.tv_sec) * 1000.0;      // sec to ms
    elapsed_time_handshake += (t2_handshake.tv_usec - t1_handshake.tv_usec) / 1000.0;   // us to ms

    elapsed_time_read = (t2_read.tv_sec - t1_read.tv_sec) * 1000.0;      // sec to ms
    elapsed_time_read += (t2_read.tv_usec - t1_read.tv_usec) / 1000.0;   // us to ms

    elapsed_time_write = (t2_write.tv_sec - t1_write.tv_sec) * 1000.0;      // sec to ms
    elapsed_time_write += (t2_write.tv_usec - t1_write.tv_usec) / 1000.0;   // us to ms

    elapsed_time_rest = (t2_rest.tv_sec - t1_rest.tv_sec) * 1000.0;      // sec to ms
    elapsed_time_rest += (t2_rest.tv_usec - t1_rest.tv_usec) / 1000.0;   // us to ms

#else
    elapsed_time = 0.0;
    elapsed_time_read = 0.0;
    elapsed_time_write = 0.0;
    elapsed_time_handshake = 0.0;
#endif

    print_table(table);
    fprintf(stderr, "Bytes written: %d\nResponse: %s\n", ret, (char *) response);

    if (strstr(response, "Inference:") != NULL) {
        fprintf(stderr, "Time to read request from client: %f ms\n", elapsed_time_read);
        fprintf(stderr, "Time to write response to client: %f ms\n", elapsed_time_write);
        fprintf(stderr, "Time to perform handshake: %f ms\n", elapsed_time_handshake);
        fprintf(stderr, "Time to process the request: %f ms\n", elapsed_time_rest);
        fprintf(stderr, "Total time - server: %f ms\n", elapsed_time);
    }

    fprintf(stderr, "Closing the connection...");

    while ((ret = mbedtls_ssl_close_notify(&ssl)) < 0) {
        if (ret != MBEDTLS_ERR_SSL_WANT_READ &&
            ret != MBEDTLS_ERR_SSL_WANT_WRITE) {
            fprintf(stderr, " failed\n   mbedtls_ssl_close_notify returned %d\n", ret);
            goto reset;
        }
    }

    fprintf(stderr, " ok\n");

    ret = 0;
    goto reset;

exit:

#ifdef MBEDTLS_ERROR_C
    if (ret != 0) {
        char error_buf[100];
        mbedtls_strerror(ret, error_buf, 100);
        fprintf(stderr, "Last error was: %d - %s\n\n", ret, error_buf);
    }
#endif

    mbedtls_net_free(&client_fd);
    mbedtls_net_free(&listen_fd);
    mbedtls_x509_crt_free(&srvcert);
    mbedtls_pk_free(&pkey);
    mbedtls_ssl_free(&ssl);
    mbedtls_ssl_config_free(&conf);
#if defined(MBEDTLS_SSL_CACHE_C)
    mbedtls_ssl_cache_free(&cache);
#endif
    mbedtls_ctr_drbg_free(&ctr_drbg);
    mbedtls_entropy_free(&entropy);
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    mbedtls_psa_crypto_free();
#endif /* MBEDTLS_USE_PSA_CRYPTO */

    exit(ret);
}
