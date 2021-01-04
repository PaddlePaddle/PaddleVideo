# Configs design

---
This page shows how PaddleVideo use the basic IOC/DI technology to decouple and control the whole framework. It is flexible to increase modularity of this system and make it extensible. At last we will explain the details of config yaml and training script args.


## Design

First, when we create a new class, it is common to new a instance like:

```python
class TSM():
    XXX

a = TSM(init_attributes)
```

when more classes are created, the coupling relationship between the calling and called method will increase sharply, obviously, we can create a factory class to solve it, like that:

```python
if model_name == "TSM":
    model = TSM()
elif model_name == "TSN":
    model = TSN()
elif ...
```
and 

```python
optimizer_cfg = dict(name:"MOMENTUM", params: XXX)
if optimizer_cfg.name = "MOMENTUM":
    optimizer = MOMENTUM(optimizer_cfg.pop(name))
elif:
    ...
```

more and more conditions have to be created though. like widly used in the Java, we apply ```inversion of control``` and ```Dependency Inversion``` to decuople.

Second, to inplenment DI, we build two components:

- Register
- Builder

1. Register

We implenment a getter and a setter function to map string to an instance.
[source code](../../paddlevideo/utils/registry.py)

```python
#excerpt from source code.
class Registry():
    def __init__(self, name):
        self._name = name
        self._obj_map = {}
        
    #mapping name -> object
    def register(self,  obj, name):
        self._obj_map[name] = obj
        
    #get object
    def get(self, name):
        ret = self._obj_map.get(name)
        return ret
```

It provides name -> object mapping. For example, To register an object:
```python
    BACKBONES = Registry('backbone') #new a Register
    @BACKBONES.register() #regist resnet as a backbone.
    class ResNet:
        pass
    Or:

    BACKBONES = Registry('backbone')
    class ResNet:
        pass
    BACKBONES.register(ResNet)
```

2. Builder
To obtain registed module.
```python
    # Usage: To build a module.
    
    backbone_name = "ResNet"
    b = BACKBONES.get(backbone_name)()
```

so that we can new(register) an instance in **where it declared**, not **where it called**, a basic DI sub-system has been created now.

We apply this design on many places, such as: PIPELINE, BACKBONE, HEAD, LOSS, METRIC

Finally, We build all of the framework components from config yaml which matches the source code one by one.

---

## config yaml details <sup>Refine soon</sup>

We separate the config to several parts, in high level:

- MODEL
- DATASET
- PIPELINE
- OPTIMIZER

and some unique global configurations, like
- model_name
- log_interval
- epochs
- resume_epoch
- log_level


training script args

-  --validate: switch validate mode on or not
-  --test: switch test mode on or not
-  --weights="": weights
-  -c: config yaml path
-  -o: override args





