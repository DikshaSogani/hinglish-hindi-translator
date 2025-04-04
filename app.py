import streamlit as st
import torch

# Load the model

@st.cache_resource
def load_model():
    checkpoint = torch.load("/content/drive/MyDrive/final_model.pth.tar", map_location=torch.device("cpu"))
    
    # Debugging: Print keys in checkpoint
    st.write("Checkpoint keys:", checkpoint.keys()) 
    
    if 'model' in checkpoint:
        model = checkpoint['model']
    else:
        st.error("Error: 'model' key not found in checkpoint. Check the saved model file.")
        return None

    model.eval()
    return model


def translate_hinglish(hinglish_text):
    try:
        input_tensor = torch.tensor([ord(c) for c in hinglish_text]).unsqueeze(0)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        translated_text = ''.join([chr(int(c)) for c in output_tensor.squeeze().tolist()])
        return translated_text
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("📝 Hinglish to Hindi Translator")
st.write("Enter Hinglish text below and get its Hindi translation.")

user_input = st.text_area("Enter Hinglish text:")
if st.button("Translate"):
    if user_input.strip():
        result = translate_hinglish(user_input)
        st.success(f"**Hindi Translation:** {result}")
    else:
        st.warning("⚠️ Please enter some text to translate.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and PyTorch")
