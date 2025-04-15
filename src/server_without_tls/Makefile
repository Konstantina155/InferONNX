USE_AES ?= 0
USE_MEMORY_ONLY ?= 0
USE_SYS_TIME_OPERATORS ?= 0

CC = gcc
CFLAGS = -Wall -Wextra -pedantic -g
LDFLAGS = -I./include -L./lib -lmbedtls -lmbedx509 -lmbedcrypto

ifeq ($(USE_MEMORY_ONLY), 1)
	CFLAGS += -DUSE_MEMORY_ONLY
endif
ifeq ($(USE_AES), 1)
    CFLAGS += -DUSE_AES
	LDFLAGS += -I./tract_aes -L./tract_aes -ltract -lm -lpthread -ldl
else
    LDFLAGS += -L./tract_no_aes
endif
all: client

client: client.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

client.o: client.c
	$(CC) $(CFLAGS) -c -o $@ $< $(LDFLAGS)

clean:
	rm -f client *.o
