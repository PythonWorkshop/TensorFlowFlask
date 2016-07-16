from flask_wtf import Form
from wtforms import DecimalField, validators

class TestParameterForm(Form):
    alcohol              = DecimalField('Alcohol Percentage')
    volatile_acidity     = DecimalField('Fixed Acidity')
    citric_acid          = DecimalField('Citric Acid')
    residual_sugar       = DecimalField('Residual Sugar')
    chlorides            = DecimalField('Chlorides')
    free_sulfur_dioxide  = DecimalField('Free Sulfur Dioxide')
    total_sulfur_dioxide = DecimalField('Total Sulfur Dioxide')
    density              = DecimalField('Density')
    ph                   = DecimalField('pH')
    sulphates            = DecimalField('Sulphates')
