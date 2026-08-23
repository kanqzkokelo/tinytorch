#ifndef ASYNC_PRINTER_H
#define ASYNC_PRINTER_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AsyncPrinter AsyncPrinter;

AsyncPrinter *async_printer_start(void);
void async_printer_push(AsyncPrinter *ap, const char *str, int len);
void async_printer_stop_and_flush(AsyncPrinter *ap);

#ifdef __cplusplus
}
#endif

#endif // ASYNC_PRINTER_H
