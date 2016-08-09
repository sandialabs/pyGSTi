# Notes on `pygsti/report/formatter.py`

### Changes:

 - Remove global variable SCRATCHDIR
 - Allow broader parameter control   
  (ex: table.render(precision=2) affects precision in **all** of a table's fields)
 - Convert functions to callable classes   
  (reduce duplicate code)
 - Stop storing formatters inside of table instances  
  (Store strings instead. Formatters no longer have to be picklable)
 - Store formatters in a static dictionary inside of FormatSet
  (Instead of globally)

### Notes:

##### `FormatSet`
The entire public interface of the formatter module is through the `FormatSet` class

All it needs is a set of specs (currently - a dictionary defining `precision`, `polarprecision`, and `scratchDir`)  
These are generally provided as keyword arguments to `table.render()`  
The only method of `FormatSet` is `formatList()`, which is called during table.render()  
This passes parameters (supplied as kwargs to `table.render()`) down to individual formatters, and eventually functions like `latex()` or `html_value()`  

##### `Adding a formatter`

A couple of helper classes exist to make this process easier, but a formatter can be anything with the signature `label -> formatter label` (It must be callable)  

For example, the following is the definition for the 'Normal' formatters

```
FormatSet.formatDict['Normal'] = {
    'html'  : _PrecisionFormatter(html),
    'latex' : _PrecisionFormatter(latex),
    'text'  : _no_format,
    'ppt'   : _PrecisionFormatter(ppt) }
```

Where `html`, `latex`, and `ppt` are functions that take the arguments `precision` and `polarprecision`  
Any function that takes these arguments can be provided to a `_PrecisionFormatter` to automatically have specs passed to it during `table.render()` calls

Here, `_no_format` is a function with the signature `l -> l` (id)

Other somewhat helpful classes are `ParameterizedFormatter` (PrecisionFormatter's parent class), and `_FigureFormatter` (A child of `ParameterizedFormatter` that utilizes a scratchDir) 



 
