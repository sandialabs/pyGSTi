{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

   .. rubric:: Sub-packages

   .. 
     This list needs to be manually updated with all
     the pygsti modules you want documented

   .. autosummary::
       :template: my_autosummary_module.rst
       :toctree:

       pygsti.algorithms       
       pygsti.construction
       pygsti.drivers
       pygsti.io
       pygsti.objects
       pygsti.optimize
       pygsti.report
       pygsti.tools


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
