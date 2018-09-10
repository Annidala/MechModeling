#coding: utf-8

# Property that is only computed once
def lazyproperty(f):
  @property
  def wrapper(self,*args,**kwargs):
    if not hasattr(self,'_'+f.__name__):
      setattr(self,'_'+f.__name__,f(self,*args,**kwargs))
    return getattr(self,'_'+f.__name__)
  return wrapper

def mysum(l): # Example function to be called in my test getter
  print("Summing...")
  return sum(l)

class Stuff():
  def __init__(self,data=[0,1,2]):
    self.data = data
    
  # s.sum will be computed on the first time we get it
  # but the same value will be returned on every other call
  # Simply delete self._sum to mark it as to be computed again
  @lazyproperty
  def sum(self):
    return mysum(self.data)

s = Stuff([2,5,8])

print("First call:")
print("sum=",s.sum)
print("second call:")
print("sum=",s.sum)
del s._sum
print("Third call:")
print("sum=",s.sum)