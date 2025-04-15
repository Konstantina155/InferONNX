USE_AES ?= 0
USE_MEMORY_ONLY ?= 0
USE_SYS_TIME_OPERATORS ?= 0

CC = gcc
CFLAGS = -Wall -Wextra -pedantic -g
LDFLAGS = -I../include -L../lib -lmbedtls -lmbedx509 -lmbedcrypto

ifeq ($(USE_MEMORY_ONLY), 1)
	CFLAGS += -DUSE_MEMORY_ONLY
endif
ifeq ($(USE_AES), 1)
    CFLAGS += -DUSE_AES
	LDFLAGS += -I../tract_aes -L../tract_aes -ltract -lm -lpthread -ldl
else
	LDFLAGS += -I../tract_no_aes -ltract -lm -lpthread -ldl
	ifeq ($(USE_SYS_TIME_OPERATORS), 1)
		LDFLAGS += -L../tract_no_aes/use_sys_time
	else
		LDFLAGS += -L../tract_no_aes/no_use_sys_time
	endif
endif

all: server

server: main.o inference.o storage.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.c
	$(CC) $(CFLAGS) -c -o $@ $< $(LDFLAGS)

inference.o: inference.c 
	$(CC) $(CFLAGS) -c -o $@ $^ $(LDFLAGS)

storage.o: storage.c
	$(CC) $(CFLAGS) -c -o $@ $< $(LDFLAGS)

clean:
	rm -f server *.o