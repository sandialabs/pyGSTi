{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

   .. rubric:: Sub-modules

   .. 
     This list needs to be manually updated with all
     the GST modules you want documented

   .. autosummary::
       :template: my_autosummary_module.rst
       :toctree:
       
       GST.AnalysisTools
       GST.BasisTools
       GST.ComputeMetrics
       GST.Core
       GST.Exceptions
       GST.Gate
       GST.GateOps
       GST.GateSetConstruction
       GST.GateSetTools
       GST.GateStringTools
       GST.GramMatrix
       GST.HtmlUtil
       GST.JamiolkowskiOps
       GST.LatexUtil
       GST.LikelihoodFunctions
       GST.ListTools
       GST.Loaders
       GST.MatrixOps
       GST.Optimize
       GST.ReportGeneration
       GST.StdInputParser
       GST.Writers

   ..  Don't document these modules separately
       GST.dataset
       GST.evaltree
       GST.gateset
       GST.gatestring
       GST.multidataset
       GST.outputdata



   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree:

   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :template: my_autosummary_class.rst
      :toctree:

   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}


   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. toctree::
   {% for item in exceptions %}
      {{ item }} <GST.{{ item }}>
   {%- endfor %}

   .. autosummary::
      :toctree:

   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
