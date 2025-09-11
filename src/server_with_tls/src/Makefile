USE_SYS_TIME ?= 0
USE_SYS_TIME_OPERATORS ?= 0
USE_OCCLUM ?= 0
USE_AES ?= 0
USE_MEMORY_ONLY ?= 0
USE_DEBUG ?= 0
NUM_TOKENS ?= 10
USE_FILE_CACHING ?= 0

CFLAGS = -Wall -Wextra -pedantic -g -std=gnu11
LDFLAGS = -I../include -L ../lib -lmbedtls -lmbedx509 -lmbedcrypto

ifeq ($(USE_OCCLUM), 1)
	CC = occlum-gcc
    CFLAGS += -DUSE_OCCLUM
else
	CC = gcc
	ifeq ($(USE_FILE_CACHING), 1)
		CFLAGS += -DUSE_FILE_CACHING
	endif
endif
ifeq ($(USE_SYS_TIME), 1)
    CFLAGS += -DUSE_SYS_TIME
endif
ifeq ($(USE_MEMORY_ONLY), 1)
    CFLAGS += -DUSE_MEMORY_ONLY
endif
ifeq ($(USE_AES), 1)
    CFLAGS += -DUSE_AES
endif
ifeq ($(USE_AES), 1)
	LDFLAGS += -I../tract_aes -ltract -lm -lpthread -ldl
	ifeq ($(USE_SYS_TIME_OPERATORS), 1)
		LDFLAGS += -L../tract_aes/use_sys_time
	else
		LDFLAGS += -L../tract_aes/no_use_sys_time
	endif
else
	LDFLAGS += -I../tract_no_aes -L../tract_no_aes -ltract -lm -lpthread -ldl
endif
ifeq ($(USE_DEBUG), 1)
    CFLAGS += -DUSE_DEBUG
endif
CFLAGS += -DNUM_TOKENS=$(NUM_TOKENS)

all: server occlum_server

server: main.o inference.o storage.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

occlum_server: occlum_main.o inference.o storage.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

occlum_main.o: occlum_main.c
	$(CC) $(CFLAGS) -c -o $@ $< $(LDFLAGS)

main.o: main.c
	$(CC) $(CFLAGS) -c -o $@ $< $(LDFLAGS)

inference.o: inference.c 
	$(CC) $(CFLAGS) -c -o $@ $^ $(LDFLAGS)

storage.o: storage.c
	$(CC) $(CFLAGS) -c -o $@ $< $(LDFLAGS)

clean:
	rm -f server occlum_server *.o
