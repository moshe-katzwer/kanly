from __future__ import absolute_import, print_function

import time

start_time = dict()


def timer(name=None, display=True, trials=None):
    """
    lightweight alternative to `timeit`. does not repeatedly execute code,
    or execute any code as all.  Shorthand for having to do
    `t=time.time(); <stuff>; print(time.time()-t);`

    Examples
    --------
    Wrap a code block with paired ``timer('label')`` calls. The first call
    starts the clock; the second prints/returns the elapsed seconds:

    >>> from kanly.api import timer
    >>> import time
    >>> timer('demo')                                # start timer
    >>> time.sleep(0.05)                             # doctest: +SKIP
    >>> elapsed = timer('demo')                      # stops + prints + returns
    name='demo', elapsed=0.050...s
    >>> elapsed > 0.04                                # doctest: +SKIP
    True

    Multiple independent timers can run concurrently via different ``name``s;
    pass ``display=False`` to suppress printing while still returning the
    elapsed seconds.
    """
    global start_time
    if name in start_time:
        elapsed = time.time() - start_time[name]
        del start_time[name]
        if display:
            if trials:
                print(f'{name=}, {elapsed/trials=:.6f}s per trial')
            else:
                print(f'{name=}, {elapsed=:.6f}s')
    else:
        start_time[name] = time.time()
        elapsed = None
    return elapsed


def clear_timers():
    global start_time
    start_time.clear()
