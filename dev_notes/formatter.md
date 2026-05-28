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

```python
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

So, for example, to define any precision formatter, I could do:

`precision_formatter = _PrecisionFormatter(f)` where f is a function with the signature `label, precision, polarprecision -> formatted label`

Since `_PrecisionFormatter` is just a helper class with `__init__` defined as:

``` python
def __init__(custom, defaults={}, formatstring='%s'):
    super(_PrecisionFormatter, self).__init__(custom, ['precision', 'polarprecision'], defaults, formatstring)
```

A precision formatter could be defined as:  
`precision_formatter = _ParameterizedFormatter(f, ['precision', 'polarprecision'])`

So, any function with the signature `label, *args -> formattedlabel` can be parameterized:  
`_ParameterizedFormatter(f, [args])`

To define a precision formatter that always rounds polars to the second place:  
`precision_formatter = _ParameterizedFormatter(f, ['precision'], {'polarprecision' : 2})`

##### Figure Formatters

All of the figure formatters:
```python
FormatSet.formatDict['Figure'] = {
    'html'  : _FigureFormatter(formatstring="<img width='%.2f' height='%.2f' src='%s/%s'>",
                               extension='.png'),
    'latex' : _FigureFormatter(formatstring="\\vcenteredhbox{\\includegraphics[width=%.2fin,height=%.2fin,keepaspectratio]{%s/%s}}",
                               extension='.pdf'),
    'text'  : lambda t : t[0], # String of figinfo
    'ppt'   : lambda t : 'ppt does not support figure formatting'} # Not Implemented
```

All of these formatters have the signature `figInfo -> formattedString (that loads an image)`

A couple of things happen here when `table.render(scratchDir='myDir')` is called:
  - `FormatSet.formatList` finds a label corresponding to the formatter `'Figure'`
  - `fig` is extracted from `figInfo`, and saved to `scratchDir` with the provided extension (default `.png`)
  - A format string (default `'%s%s%s%s'`) is provided `W, H, scratchDir, filename+extension` (W, H are extracted from figInfo as well)
  - The formatstring is returned (ex `formatstring="<img width='10.0' height='10.0' src='myDir/myFig.png'>"`)


 
