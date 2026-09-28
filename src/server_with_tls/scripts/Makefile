USE_MEMORY_ONLY ?= 0
CACHE_STATS_RUNS ?= 1
NUM_TOKENS ?= 10
USE_PII_REGEX ?= 0
USE_PII_GUARDRAILS ?= 0

CFLAGS = -Wall -Wextra -pedantic -g -DUSE_PII_GUARDRAILS=$(USE_PII_GUARDRAILS)
#LDFLAGS = -I../tract_no_aes/ -L../tract_no_aes/ -ltract -lpthread -lm -ldl

ifeq ($(USE_MEMORY_ONLY), 1)
    CFLAGS += -DUSE_MEMORY_ONLY
endif
ifeq ($(USE_PII_REGEX), 1)
    CFLAGS += -DUSE_PII_REGEX
	LDFLAGS = -I../../../../../tract/api/c/use_pii_regex/ -L../../../../../tract/api/c/use_pii_regex/ -ltract -lpthread -lm -ldl
else ifeq ($(USE_PII_GUARDRAILS), 0)
	LDFLAGS = -I../../../../../tract/api/c/ -L../../../../../tract/api/c/ -ltract -lpthread -lm -ldl
else
	LDFLAGS = -I../../../../../tract/api/c/no_use_pii_regex/ -L../../../../../tract/api/c/no_use_pii_regex/ -ltract -lpthread -lm -ldl
endif
CFLAGS += -DCACHE_STATS_RUNS=$(CACHE_STATS_RUNS) -DNUM_TOKENS=$(NUM_TOKENS)

all: standalone_inference

standalone_inference:
	gcc $(CFLAGS) standalone_inference.c -o $@ $(LDFLAGS)

clean:
	rm -f standalone_inference
