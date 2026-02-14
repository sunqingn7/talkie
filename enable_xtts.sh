#!/bin/bash
# Script to enable XTTS-v2 multilingual TTS model
# This model supports 17 languages including English and Chinese

echo "🌍 Enabling XTTS-v2 Multilingual TTS Model"
echo "=========================================="
echo ""
echo "This will enable XTTS-v2 which supports:"
echo "  - English 🇬🇧"
echo "  - Chinese (中文) 🇨🇳"
echo "  - Spanish 🇪🇸"
echo "  - French 🇫🇷"
echo "  - German 🇩🇪"
echo "  - Japanese 🇯🇵"
echo "  - Korean 🇰🇷"
echo "  - Arabic 🇸🇦"
echo "  - And 9 more languages!"
echo ""
echo "⚠️  IMPORTANT: You must accept the Coqui license to use XTTS-v2"
echo "   License: CPML (non-commercial) unless you purchase a commercial license"
echo "   More info: https://coqui.ai/cpml"
echo ""

# Accept license
export COQUI_TOS_AGREED=1

# Update config file to use XTTS-v2
CONFIG_FILE="config/settings.yaml"

# Backup original config
cp $CONFIG_FILE ${CONFIG_FILE}.backup

# Comment out current model line and uncomment XTTS-v2 line
sed -i 's/^  coqui_model: "tts_models\/en\/ljspeech\/tacotron2-DDC"/#  coqui_model: "tts_models\/en\/ljspeech\/tacotron2-DDC"/' $CONFIG_FILE
sed -i 's/^  # coqui_model: "tts_models\/multilingual\/multi-dataset\/xtts_v2"/  coqui_model: "tts_models\/multilingual\/multi-dataset\/xtts_v2"/' $CONFIG_FILE

echo "✅ Configuration updated!"
echo ""
echo "📝 To start using XTTS-v2, run:"
echo "   export COQUI_TOS_AGREED=1"
echo "   python src/main.py"
echo ""
echo "💾 Configuration backup saved to: ${CONFIG_FILE}.backup"
