{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

   .. rubric:: Sub-modules

   .. 
     This list needs to be manually updated with all
     the GSTCommons modules you want documented

   .. autosummary::
       :template: my_autosummary_module.rst
       :toctree:
       
       GSTCommons.Analyze_LasGermExponent
       GSTCommons.Analyze_TruncatedGermPowers
       GSTCommons.Analyze_WholeGermPowers
       GSTCommons.FiducialPairReduction
       GSTCommons.MakeLists_LasGermExponent
       GSTCommons.MakeLists_TruncatedGermPowers
       GSTCommons.MakeLists_WholeGermPowers
       GSTCommons.Std1Q_XYI
       GSTCommons.Std1Q_XY


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
