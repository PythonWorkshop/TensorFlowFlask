from flask_wtf import Form
from wtforms import DecimalField, FileField, StringField
from wtforms.validators import DataRequired


class TestParameterForm(Form):
    alcohol = DecimalField('Alcohol Percentage')
    volatile_acidity = DecimalField('Fixed Acidity')
    citric_acid = DecimalField('Citric Acid')
    residual_sugar = DecimalField('Residual Sugar')
    chlorides = DecimalField('Chlorides')
    free_sulfur_dioxide = DecimalField('Free Sulfur Dioxide')
    total_sulfur_dioxide = DecimalField('Total Sulfur Dioxide')
    density = DecimalField('Density')
    ph = DecimalField('pH')
    sulphates = DecimalField('Sulphates')


class TrainingDataForm(Form):
    training_data = FileField('Training Data CSV File')
    learning_rate = DecimalField('Learning Rate')
    batch_size = DecimalField('Batch Size')
    model_name = StringField('Model Name', validators=[DataRequired()])
