/* mmap() replacement for Windows
 *
 * Author: Mike Frysinger <vapier@gentoo.org>
 * Placed into the public domain
 */

/* References:
 * CreateFileMapping: http://msdn.microsoft.com/en-us/library/aa366537(VS.85).aspx
 * CloseHandle:       http://msdn.microsoft.com/en-us/library/ms724211(VS.85).aspx
 * MapViewOfFile:     http://msdn.microsoft.com/en-us/library/aa366761(VS.85).aspx
 * UnmapViewOfFile:   http://msdn.microsoft.com/en-us/library/aa366882(VS.85).aspx
 */

#include <io.h>
#include <windows.h>
#include <sys/types.h>

#define PROT_READ     0x1
#define PROT_WRITE    0x2
/* This flag is only available in WinXP+ */
#ifdef FILE_MAP_EXECUTE
#define PROT_EXEC     0x4
#else
#define PROT_EXEC        0x0
#define FILE_MAP_EXECUTE 0
#endif

#define MAP_SHARED    0x01
#define MAP_PRIVATE   0x02
#define MAP_ANONYMOUS 0x20
#define MAP_ANON      MAP_ANONYMOUS
#define MAP_FAILED    ((void *) -1)

#ifdef __USE_FILE_OFFSET64
# define DWORD_HI(x) (x >> 32)
# define DWORD_LO(x) ((x) & 0xffffffff)
#else
# define DWORD_HI(x) (0)
# define DWORD_LO(x) (x)
#endif

static void *mmap(void *start, size_t length, int prot, int flags, int fd, off_t offset)
{
    if (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC))
        return MAP_FAILED;
    if (fd == -1) {
        if (!(flags & MAP_ANON) || offset)
            return MAP_FAILED;
    } else if (flags & MAP_ANON)
        return MAP_FAILED;

    DWORD flProtect;
    if (prot & PROT_WRITE) {
        if (prot & PROT_EXEC)
            flProtect = PAGE_EXECUTE_READWRITE;
        else
            flProtect = PAGE_READWRITE;
    } else if (prot & PROT_EXEC) {
        if (prot & PROT_READ)
            flProtect = PAGE_EXECUTE_READ;
        else if (prot & PROT_EXEC)
            flProtect = PAGE_EXECUTE;
    } else
        flProtect = PAGE_READONLY;

    off_t end = length + offset;
    HANDLE mmap_fd, h;
    if (fd == -1)
        mmap_fd = INVALID_HANDLE_VALUE;
    else
        mmap_fd = (HANDLE)_get_osfhandle(fd);
    h = CreateFileMapping(mmap_fd, NULL, flProtect, DWORD_HI(end), DWORD_LO(end), NULL);
    if (h == NULL)
        return MAP_FAILED;

    DWORD dwDesiredAccess;
    if (prot & PROT_WRITE)
        dwDesiredAccess = FILE_MAP_WRITE;
    else
        dwDesiredAccess = FILE_MAP_READ;
    if (prot & PROT_EXEC)
        dwDesiredAccess |= FILE_MAP_EXECUTE;
    if (flags & MAP_PRIVATE)
        dwDesiredAccess |= FILE_MAP_COPY;
    void *ret = MapViewOfFile(h, dwDesiredAccess, DWORD_HI(offset), DWORD_LO(offset), length);
    if (ret == NULL) {
        CloseHandle(h);
        ret = MAP_FAILED;
    }
    return ret;
}

static void munmap(void *addr, size_t length)
{
    UnmapViewOfFile(addr);
    /* ruh-ro, we leaked handle from CreateFileMapping() ... */
}

#undef DWORD_HI
#undef DWORD_LO
