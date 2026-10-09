USE_MEMORY_ONLY ?= 0
CACHE_STATS_RUNS ?= 1
NUM_TOKENS ?= 10

CFLAGS = -Wall -Wextra -pedantic -g
LDFLAGS = -I../tract_no_aes/ -L../tract_no_aes/ -ltract -lpthread -lm -ldl

ifeq ($(USE_MEMORY_ONLY), 1)
    CFLAGS += -DUSE_MEMORY_ONLY
endif

all: onnx_to_nnef_conversion standalone_inference

onnx_to_nnef_conversion:
	gcc $(CFLAGS) onnx_to_nnef_conversion.c -o $@ $(LDFLAGS)

standalone_inference: CFLAGS += -DCACHE_STATS_RUNS=$(CACHE_STATS_RUNS) -DNUM_TOKENS=$(NUM_TOKENS)
standalone_inference:
	gcc $(CFLAGS) standalone_inference.c -o $@ $(LDFLAGS)

clean:
	rm -f standalone_inference onnx_to_nnef_conversion
