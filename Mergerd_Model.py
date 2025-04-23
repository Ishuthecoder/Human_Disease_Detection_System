from keras.models import load_model, Model
from keras.layers import Input, Concatenate, Dense

# Load all five models
base_model1 = load_model("Multi_Cancer_Model1.keras", compile=False)
base_model2 = load_model("Covid_Model_Infection.keras", compile=False)
base_model3 = load_model("Covid_Model_Segmentation.keras", compile=False)
base_model4 = load_model("ISIC_deseases.keras", compile=False)
base_model5 = load_model("Completed_project.keras", compile=False)

print("✅ All 5 models loaded successfully!")

# Freeze all layers
for model in [base_model1, base_model2, base_model3, base_model4, base_model5]:
    for layer in model.layers:
        layer.trainable = False

# Wrap models with unique names
model1 = Model(inputs=base_model1.input, outputs=base_model1.output, name="Model1_MultiCancer")
model2 = Model(inputs=base_model2.input, outputs=base_model2.output, name="Model2_CovidInfection")
model3 = Model(inputs=base_model3.input, outputs=base_model3.output, name="Model3_CovidSegmentation")
model4 = Model(inputs=base_model4.input, outputs=base_model4.output, name="Model4_ISIC")
model5 = Model(inputs=base_model5.input, outputs=base_model5.output, name="Model5_Completed")

# Use input shape from the first model
input_shape = base_model1.input_shape[1:]
common_input = Input(shape=input_shape)

# Get outputs from each sub-model
output1 = model1(common_input)
output2 = model2(common_input)
output3 = model3(common_input)
output4 = model4(common_input)
output5 = model5(common_input)

# Merge outputs
merged = Concatenate()([output1, output2, output3, output4, output5])

# Add final classification layer (adjust units as per your use-case)
final_output = Dense(3, activation='softmax', name='final_output')(merged)

# Create unified model
unified_model = Model(inputs=common_input, outputs=final_output)

# Show summary
unified_model.summary()

# Save the unified model
unified_model.save("Unified_Five_Models.keras")

print("✅ Unified model saved as 'Unified_Five_Models.keras'")
