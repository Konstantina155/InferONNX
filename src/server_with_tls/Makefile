USE_SYS_TIME ?= 0
USE_SYS_TIME_OPERATORS ?= 0
USE_OCCLUM ?= 0
USE_AES ?= 0
USE_MEMORY_ONLY ?= 0
USE_STRIP ?= 0

CC = gcc
CFLAGS = -Wall -Wextra -pedantic -g
LDFLAGS = -I./include -L ./lib -lmbedtls -lmbedx509 -lmbedcrypto

ifeq ($(USE_STRIP), 1)
	CFLAGS += -DUSE_STRIP
endif
ifeq ($(USE_SYS_TIME), 1)
    CFLAGS += -DUSE_SYS_TIME
endif
ifeq ($(USE_OCCLUM), 1)
    CFLAGS += -DUSE_OCCLUM
endif
ifeq ($(USE_MEMORY_ONLY), 1)
    CFLAGS += -DUSE_MEMORY_ONLY
endif
ifeq ($(USE_AES), 1)
    CFLAGS += -DUSE_AES
    LDFLAGS += -I./tract_aes -ltract -lm -lpthread -ldl
    ifeq ($(USE_SYS_TIME_OPERATORS), 1)
    	LDFLAGS += -L./tract_aes/use_sys_time
    else
        LDFLAGS += -L./tract_aes/no_use_sys_time
    endif
else
	LDFLAGS += -I./tract_no_aes -L./tract_no_aes -ltract -lm -lpthread
endif

all: ssl_client

ssl_client: ssl_client.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ssl_client.o: ssl_client.c
	$(CC) $(CFLAGS) -c -o $@ $< $(LDFLAGS)

clean:
	rm -f ssl_client *.o
