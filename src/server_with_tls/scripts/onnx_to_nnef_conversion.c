#include <stdio.h>
#include <stdlib.h>
#include <tract.h>
#include <string.h>

#define check(call) {                                                           \
    TRACT_RESULT result = call;                                                 \
    if(result == TRACT_RESULT_KO) {                                             \
        fprintf(stderr, "Error calling tract: %s\n", tract_get_last_error());   \
        exit(1) ;                                                               \
    }                                                                           \
}

int
main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <onnx_model_path> <nnef_model_path>\n", argv[0]);
        return 1;
    }

    const char *onnx_model_path = argv[1];
    const char *nnef_model_path = argv[2];

    printf(
        "Converting ONNX -> NNEF:\n"
        "  input : %s\n"
        "  output: %s\n",
        onnx_model_path,
        nnef_model_path
    );

    check(convert_onnx_file_to_nnef(onnx_model_path, nnef_model_path));

    return 0;
}