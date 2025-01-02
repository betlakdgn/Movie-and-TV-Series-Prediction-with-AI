from src.predict import predict_genre  # Tahmin fonksiyonunu içe aktar

if __name__ == "__main__":
    while True:
        user_input = input("Film açıklamasını girin (çıkmak için 'exit' yazın): ")
        if user_input.lower() == "exit":
            print("Çıkılıyor...")
            break

        # Tahmin yap
        try:
            predicted_genre = predict_genre(user_input)
            print(f"Tahmin edilen tür: {predicted_genre}")
        except Exception as e:
            print(f"Hata oluştu: {e}")
