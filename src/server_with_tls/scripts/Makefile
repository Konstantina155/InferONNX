USE_MEMORY_ONLY ?= 0
CACHE_STATS_RUNS ?= 1
NUM_TOKENS ?= 10

CFLAGS = -Wall -Wextra -pedantic -g
#LDFLAGS = -I../tract_no_aes/ -L../tract_no_aes/ -ltract -lpthread -lm -ldl
LDFLAGS = -I../../../../../tract/api/c -L../../../../../tract/api/c -ltract -lpthread -lm -ldl
#LDFLAGS = -I../../../../../simple_optimized/ -L../../../../../simple_optimized/ -ltract -lpthread -lm -ldl

ifeq ($(USE_MEMORY_ONLY), 1)
    CFLAGS += -DUSE_MEMORY_ONLY
endif
CFLAGS += -DCACHE_STATS_RUNS=$(CACHE_STATS_RUNS) -DNUM_TOKENS=$(NUM_TOKENS)

all: standalone_inference

standalone_inference:
	gcc $(CFLAGS) standalone_inference.c -o $@ $(LDFLAGS)

clean:
	rm -f standalone_inference
