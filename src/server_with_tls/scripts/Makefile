USE_MEMORY_ONLY ?= 0
CACHE_STATS_RUNS ?= 1

CFLAGS = -Wall -Wextra -pedantic -g
LDFLAGS = -I../tract_no_aes -L../tract_no_aes -ltract -lpthread -lm -ldl

ifeq ($(USE_MEMORY_ONLY), 1)
    CFLAGS += -USE_MEMORY_ONLY
endif
CFLAGS += -DCACHE_STATS_RUNS=$(CACHE_STATS_RUNS)

all: standalone_inference

standalone_inference:
	gcc $(CFLAGS) standalone_inference.c -o $@ $(LDFLAGS)

clean:
	rm -f standalone_inference
