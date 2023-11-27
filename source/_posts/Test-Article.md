---
title: Test Article
date: 2023-10-28 21:08:10
tags:
---

# Hi there!

如果你看到这个，说明这个博客框架似乎运行正常。
This marks the blog is running well.

下面的部分用于测试hexo和stellar主题扩展的相关功能，请忽略其中的内容。

### 表情
{% emoji tieba 滑稽 %}
{% emoji 爱你 %}
{% emoji blobcat ablobcatrainbow %}
{% emoji blobcat ablobcatattentionreverse %}

### 行内文本修饰

#### 美化文本格式
- 这是 {% psw 你知道的太多了 %} 隐藏标签
- 这是 {% u 下划线 %} 标签
- 这是 {% emp 着重号 %} 标签
- 这是 {% wavy 波浪线 %} 标签
- 这是 {% del 删除线 %} 标签
- 这是 X{% sup 2 color:red %} 上标标签
- 这是 X{% sub 2 %} 下标标签
- 这是 {% kbd 键盘样式 %} 标签，试一试：{% kbd Ctrl %} + {% kbd D %}

#### mark 标记
- 这是彩色标记：{% mark 默认 %} {% mark 红 color:red %} {% mark 橙 color:orange %} {% mark 黄 color:yellow %} {% mark 绿 color:green %} {% mark 青 color:cyan %} {% mark 蓝 color:blue %} {% mark 紫 color:purple %} 
- 这是底色标记：{% mark 浅 color:light %} {% mark 深 color:dark %} 
- 这是特殊标记： {% mark 警告 color:warning %} {% mark 错误 color:error %} 
- 一共 12 种颜色。


### tag标签
和mark类似，但是可以高亮和添加链接，使用属性color指定颜色，不指定为随机
{% tag Stellar https://xaoxuu.com/wiki/stellar/ %}
{% tag Hexo https://hexo.io/ %}
{% tag MyGitHub https://github.com/Nick-bit233 color:blue %}

### 使用标签插入图片
这将比使用默认markdown格式插入图片要好
- src: 图片地址
- description: 图片描述
- width:和padding:可以对不同的尺寸做适应
- bg:可以添加背景颜色，使用bg:var(--card)使得背景颜色适配全局配色
- download: 设置为true可以增加一个下载图片的按钮，但这只会从src中下载图片，如果（对于大图）图片的下载地址和src的预览地址不同，可以将下载地址写在这里
- 点击放大功能：在任意 `image` 标签中增加 `fancybox:true` 参数即可为特定图片开启缩放功能

```
{% image src [description] [download:bool/string] [width:px] [padding:px] [bg:hex] %}
```

{% image https://byr.pt/ckfinder/userfiles/images/27d8962e-f3e4-43e9-a9e7-9d3086bbc424.png 点击下载你可以获得另一张来自Apple的图片 download:https://www.apple.com.cn/newsroom/images/product/iphone/lifestyle/Apple_ShotoniPhone_pieter_de_vries_011221.zip %}

### 文本美化块

#### 美化的引用
- 适合居中且醒目的引用：{% quot 这句话不是我说的 ——鲁迅 %}
- 支持自定义引号：{% quot 话题001 icon:hashtag %}

#### 诗词文本展示
{% poetry 游山西村 author:陆游 footer:诗词节选 %}
莫笑农家腊酒浑，丰年留客足鸡豚。
**山重水复疑无路，柳暗花明又一村。**
箫鼓追随春社近，衣冠简朴古风存。
从今若许闲乘月，拄杖无时夜叩门。
{% endpoetry %}


#### 备注块
正式内容中，第一个空格前面的是标题，后面的是正文。如果标题中需要显示空格，请使用`&nbsp;`代替。
- 无色备注块：和代码块一样的展示方式
{% note 你&nbsp;标题&nbsp;中有&nbsp;空&nbsp;格 正文部分可以随便空格。 %}
- 彩色备注块：使用`color:<color>`设置背景颜色，颜色标签和行内文本一致
{% note color:cyan 可用颜色： color = red、orange、yellow、green、cyan、blue、purple、light、dark、warning、error %}
- 只使用标题的备注块，使得文本默认加粗：
{% note color:yellow 这个只有标题，没有正文。 %}

### 外链卡片
```
{% link href [title] [icon:src] [desc:true/false] %}
href: 链接
title: 可选，手动设置标题（为空时会自动抓取页面标题）
icon: 可选，手动设置图标（为空时会自动抓取页面图标）
desc: 可选，是否显示摘要描述，为true时将会显示页面描述
```

{% link https://www.bilibili.com/video/BV1ic411R77s/ Nickbit的bilibili投稿 desc:true %}

### mermaid代码语言绘图

> 需要安装插件，目前还没有支持

使用前需要在 Markdown 文件开头加入
```markdown
---
mermaid: true
---
```
然后使用代码块，在语言中指定`mermaid`即可

### frame插入一个移动设备UI框架
- 可用插选择参数img（图片）和video（视频）

{% frame iphone11 img:http://arcwiki.mcd.blue/images/thumb/a/ac/Songs_mazymetroplex.jpg/256px-Songs_mazymetroplex.jpg focus:top %}


### iframe插入视频
<iframe src="//player.bilibili.com/player.html?aid=280702189&bvid=BV1ic411R77s&cid=1324970577&p=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>


### 静态时间线

{% timeline %}
<!-- node 2021 年 6 月 -->
BT夏季版banner
{% image https://byr.pt/styles/banners/2023-07-20_22-47-30.png width:300px %}
<!-- node 2021 年 2 月 -->
过年放假
{% endtimeline %}


### 友情链接

要现在`source/_data/links.yml`中加入链接的静态数据：
```yaml
'链接名称':
    - title: 某某某
      url: https://
      screenshot:
      avatar:
      description:
```
然后这样写：
```markdown
{% friends 链接名称 %}
```

### 网站卡片链接

同样要先在`source/_data/links.yml`中加入链接的静态数据，写法和友链一样。
写法是：
```markdown
{% sites 分组名 %}
```

{% sites 616sb %}
{% sites bestdori %}


### github card卡片

填写github仓库的名称后缀即可
{% ghcard Nick-bit233/Apex-legend-audio-extracter theme:dark %}

### 容器标签
支持更加丰富的分块文本，note等标签是由容器标签简化实现的。
容器标签需要多行来写

- 标准容器块
{% ablock 这是标题 color:warning %}
这是容器块中的内容
插入一个链接：[#172](https://github.com/volantis-x/hexo-theme-volantis/issues/712)
{% endablock %}

- 可折叠标题的容器块
容器块中可用嵌套其他块，包括另一个容器块。使用child属性设置嵌入块的属性
open属性设置块是否默认打开
{% folding child:codeblock open:true color:yellow 默认打开的代码折叠框 %}
```代码块```
{% endfolding %}

- 平铺折叠列表块

{% folders %}
<!-- folder #1 -->
这是答案1
<!-- folder #2 -->
这是答案2
<!-- folder #3 -->
这是答案3
{% endfolders %}

- 分栏tab容器

方便地切换展示的内容，也可以嵌套
{% tabs active:2 align:center %}

<!-- tab 图片 -->
{% image http://arcwiki.mcd.blue/images/thumb/c/c7/Songs_tsukinimurakumo.jpg/256px-Songs_tsukinimurakumo.jpg width:300px %}

<!-- tab 代码块 -->
```swift
let x = 123
print("hello world")
```

<!-- tab 表格 -->
| a | b | c |
| --- | --- | --- |
| a1 | b1 | c1 |
| a2 | b2 | c2 |

{% endtabs %}

- 轮播容器
适用于轮播图片，默认一张图片是 50% 宽度，通过设置 width:min 设置为 25% 宽度，width:max 设置为 100% 宽度。

{% swiper effect:cards %}
![](https://images.unsplash.com/photo-1625171515821-1870deb2743b?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=774&q=80)
![](https://images.unsplash.com/photo-1528283648649-33347faa5d9e?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=774&q=80)
![](https://images.unsplash.com/photo-1542272201-b1ca555f8505?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=774&q=80)
![](https://images.unsplash.com/photo-1524797905120-92940d3a18d6?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=774&q=80)
{% endswiper %}