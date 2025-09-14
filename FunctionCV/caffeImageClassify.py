def classify(img):
    image = img.copy()
    blob = cv2.dnn.blobFromImage(image = image, scalefactor = 0.017, size = (224, 224), mean = (104, 117, 123))
    model.setInput(blob)
    outputs = model.forward()
    final_output = outputs[0]
    final_output = final_output.reshape(1000,1)
    label_id = np.argmax(final_output)
    probs = np.exp(final_output) / np.sum(np.exp(final_output))
    final_prob = np.max(probs) * 100
    out_name = class_names[label_id]
    out_text = f'{out_name} : {final_prob}'
    return out_texts