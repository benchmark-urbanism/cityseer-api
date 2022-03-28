---
layout: '../layouts/PostLayout.astro'
---

<div class="yap module">
  <h1 class="yap module-title" id="tests-mock-file">tests.mock_file</h1>
  <div class="yap docstring">module docstring content more content</div>
  <section class="yap func">
    <h2 class="yap func-title" id="mock-function">mock_function</h2>
    <div class="yap func-sig-content">
      <div class="yap func-sig-title">
        <div class="yap func-sig-start">mock_function(</div>
        <div class="yap func-sig-params">
          <div class="yap func-sig-param">param_a, </div>
          <div class="yap func-sig-param">param_b=2</div>
        </div>
        <div class="yap func-sig-end">)</div>
      </div>
    </div>
    <div class="yap func-doc-str-content">
      <div class="yap doc-str-content">
        <p class="yap doc-str-text">A mock function returning a sum of param_a and param_b if positive numbers, else None</p>
        <h2 class="yap doc-str-heading">Parameters</h2>
        <div class="yap doc-str-elem-container">
          <div class="yap doc-str-elem-def">
            <div class="yap doc-str-elem-name">param_a</div>
            <div class="yap doc-str-elem-type">int</div>
          </div>
          <div class="yap doc-str-elem-desc">A *test* _param_.</div>
        </div>
        <div class="yap doc-str-elem-container">
          <div class="yap doc-str-elem-def">
            <div class="yap doc-str-elem-name">param_b</div>
            <div class="yap doc-str-elem-type">int | float</div>
          </div>
          <div class="yap doc-str-elem-desc">Another *test* _param_.

| col A |: col B |
|=======|========|
| boo | baa |</div>

</div>
<h2 class="yap doc-str-heading">Returns</h2>
<div class="yap doc-str-elem-container">
<div class="yap doc-str-elem-def">
<div class="yap doc-str-elem-name">summed*number</div>
<div class="yap doc-str-elem-type">int | float</div>
</div>
<div class="yap doc-str-elem-desc">The sum of \_param_a* and _param_b_.</div>
</div>
<div class="yap doc-str-elem-container">
<div class="yap doc-str-elem-def">
<div class="yap doc-str-elem-name"></div>
<div class="yap doc-str-elem-type">None</div>
</div>
<div class="yap doc-str-elem-desc">None returned if values are negative.</div>
</div>
<h2 class="yap doc-str-heading">Raises</h2>
<div class="yap doc-str-elem-container">
<div class="yap doc-str-elem-def">
<div class="yap doc-str-elem-name"></div>
<div class="yap doc-str-elem-type">ValueError</div>
</div>
<div class="yap doc-str-elem-desc">Raises value error if params are not numbers.</div>
</div>
<div class="yap doc-str-meta">
<h2 class="yap doc-str-heading">Notes</h2>
<p class="yap doc-str-meta-desc">
```python
print(mock_function(1, 2))

# prints 3

```

Random text

_Random table_

| col A |: col B |
|=======|========|
| boo   | baa    |
</p>
        </div>
      </div>
    </div>
  </section>
  <section class="yap class">
    <h2 class="yap class-title" id="parentclass">ParentClass</h2>
    <div class="yap class-doc-str-content">A parent class</div>
    <div class="yap class-prop-def">
      <div class="yap class-prop-def-name">parent_prop</div>
      <div class="yap class-prop-def-type">str</div>
    </div>
    <section class="yap func">
      <h2 class="yap func-title" id="parentclass">ParentClass</h2>
      <div class="yap func-sig-content">
        <div class="yap func-sig-title">
          <div class="yap func-sig-start">ParentClass(</div>
          <div class="yap func-sig-params">
            <div class="yap func-sig-param">**kwargs</div>
          </div>
          <div class="yap func-sig-end">)</div>
        </div>
      </div>
      <div class="yap func-doc-str-content">
        <div class="yap doc-str-content">
          <p class="yap doc-str-text">Parent initialisation.</p>
          <h2 class="yap doc-str-heading">Parameters</h2>
          <div class="yap doc-str-elem-container">
            <div class="yap doc-str-elem-def">
              <div class="yap doc-str-elem-name">**kwargs</div>
              <div class="yap doc-str-elem-type"></div>
            </div>
            <div class="yap doc-str-elem-desc">Keyword args.</div>
          </div>
        </div>
      </div>
    </section>
  </section>
  <section class="yap class">
    <h2 class="yap class-title" id="childclass">ChildClass</h2>
    <div class="yap class-doc-str-content">A child class</div>
    <section class="yap func">
      <h2 class="yap func-title" id="childclass">ChildClass</h2>
      <div class="yap func-sig-content">
        <div class="yap func-sig-title">
          <div class="yap func-sig-start">ChildClass(</div>
          <div class="yap func-sig-params">
            <div class="yap func-sig-param">param_c=1.1, </div>
            <div class="yap func-sig-param">param_d=0.9, </div>
            <div class="yap func-sig-param">**kwargs</div>
          </div>
          <div class="yap func-sig-end">)</div>
        </div>
      </div>
      <div class="yap func-doc-str-content">
        <div class="yap doc-str-content">
          <p class="yap doc-str-text">Child initialisation.</p>
          <h2 class="yap doc-str-heading">Parameters</h2>
          <div class="yap doc-str-elem-container">
            <div class="yap doc-str-elem-def">
              <div class="yap doc-str-elem-name">param_c</div>
              <div class="yap doc-str-elem-type">float</div>
            </div>
            <div class="yap doc-str-elem-desc">Yet another test param.</div>
          </div>
          <div class="yap doc-str-elem-container">
            <div class="yap doc-str-elem-def">
              <div class="yap doc-str-elem-name">param_d</div>
              <div class="yap doc-str-elem-type">float</div>
            </div>
            <div class="yap doc-str-elem-desc">And another.</div>
          </div>
          <div class="yap doc-str-elem-container">
            <div class="yap doc-str-elem-def">
              <div class="yap doc-str-elem-name">**kwargs</div>
              <div class="yap doc-str-elem-type"></div>
            </div>
            <div class="yap doc-str-elem-desc">Keyword args.</div>
          </div>
        </div>
      </div>
    </section>
    <div class="yap class-prop">
      <div class="yap class-prop-name">param_e</div>
      <div class="yap class-prop-type"></div>
    </div>
    <section class="yap func">
      <h2 class="yap func-title" id="childclass-hello">ChildClass.hello</h2>
      <div class="yap func-sig-content">
        <div class="yap func-sig-title">
          <div class="yap func-sig-start">ChildClass.hello(</div>
          <div class="yap func-sig-params"></div>
          <div class="yap func-sig-end">)</div>
        </div>
      </div>
      <div class="yap func-doc-str-content">
        <div class="yap doc-str-content">
          <p class="yap doc-str-text">A random class method returning &quot;hello&quot;</p>
          <h2 class="yap doc-str-heading">Returns</h2>
          <div class="yap doc-str-elem-container">
            <div class="yap doc-str-elem-def">
              <div class="yap doc-str-elem-name">str</div>
              <div class="yap doc-str-elem-type">saying_hello</div>
            </div>
            <div class="yap doc-str-elem-desc">A string saying &quot;hello&quot;</div>
          </div>
        </div>
      </div>
    </section>
  </section>
</div>
```
