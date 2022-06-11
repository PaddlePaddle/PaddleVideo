# <center>Element(DOM) drag and resize (drag拖拽缩放插件)</center>

> Dragable is an element drag-and-zoom plug-in. It can freely implement drag-and-zoom, zoom and arrange layout of DOM elements with high scalability

> draggable是一个元素拖拽缩放插件，它可以自由的实现DOM元素的拖拽、缩放、排列布局，可扩展性高

### 1.references in script tags(在script标签中引入)

```html

<script src="drag.js"></script>
```

### 2.how to use(如何使用)?

```javascript
new Draggable(container, elem, dragHandle, resizeHandle, isPixel, dragFn, resizeFn)
```

##### params introduce(参数介绍)

- container

> Outer container for dragging elements(拖拽元素的外层容器)

- elem

> Dragged Zoom Elements(被拖拽缩放的元素)

- dragHandle

> Drag handle(拖拽的手柄)

- resizeHandle

> Zoom handle(缩放的手柄)

- isPixel

> Whether the element is positioned according to the pixel, default is true, false is percentage positioning(元素是否按照像素定位，默认为true,false则为百分比定位)

- dragFn

> Callback function after dragging(拖拽之后的回调函数)

- resizeFn

> Scaled callback function(缩放之后的回调函数)
