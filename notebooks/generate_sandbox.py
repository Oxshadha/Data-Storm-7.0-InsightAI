import nbformat as nbf
# Using the nb object generated above
with open('notebooks/05_Parameter_Tuning_Sandbox.ipynb', 'w') as f_out:
    nbf.write(nb, f_out)
print("Generated 05_Parameter_Tuning_Sandbox.ipynb")
