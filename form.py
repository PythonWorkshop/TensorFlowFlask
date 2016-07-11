from flask_wtf import Form
from wtforms import DecimalField, validators

class TestParameterForm(Form):
    alcohol_percentage = DecimalField('Alcohol Percentage')
    fixed_acidity = DecimalField('Fixed Acidity')
