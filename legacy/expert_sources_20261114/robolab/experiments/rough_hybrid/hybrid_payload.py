"""Absolute-update payload curriculum, independent of group boundaries."""
def payload_range(update):
    if type(update) is not int or update < 0:raise ValueError('nonnegative absolute update required')
    if update < 1000:return (.1,.5)
    if update < 3000:return (.1,.5+14.5*((update-1000)//500+1)/4)
    return (.1,15.)

def payload_segments(first,stop):
    if not 0 <= first < stop:raise ValueError('invalid update interval')
    for end in sorted(set(x for x in (1000,1500,2000,2500,3000,stop) if first<x<=stop)):
        yield first,end,payload_range(first)
        first=end
