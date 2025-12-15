from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 存储审阅数据的文件路径
DATA_FILE = 'review_data.json'

# 确保存储目录存在
os.makedirs(os.path.dirname(DATA_FILE) if os.path.dirname(DATA_FILE) else '.', exist_ok=True)

# 确保存储文件存在
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)

def load_review_data():
    """加载审阅数据"""
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def save_review_data(data):
    """保存审阅数据"""
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

@app.route('/api/review', methods=['POST'])
def receive_review():
    """接收审阅数据"""
    try:
        # 获取JSON数据
        review_data = request.get_json()
        
        if not review_data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
        
        # 验证必要字段
        if 'original' not in review_data or 'modified' not in review_data:
            return jsonify({'success': False, 'error': 'Missing required fields: original and modified'}), 400
        
        # 添加时间戳
        review_data['timestamp'] = datetime.now().isoformat()
        
        # 加载现有数据
        all_reviews = load_review_data()
        
        # 添加新数据
        all_reviews.append(review_data)
        
        # 保存数据
        if save_review_data(all_reviews):
            print(f"Received review data: {review_data}")
            return jsonify({'success': True}), 200
        else:
            return jsonify({'success': False, 'error': 'Failed to save data'}), 500
            
    except Exception as e:
        print(f"Error processing review: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reviews', methods=['GET'])
def get_reviews():
    """获取所有审阅数据"""
    try:
        reviews = load_review_data()
        return jsonify({'success': True, 'data': reviews}), 200
    except Exception as e:
        print(f"Error retrieving reviews: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """获取数据分析结果"""
    try:
        reviews = load_review_data()
        
        # 基本统计
        total_reviews = len(reviews)
        
        # 分析修改模式
        modifications = []
        total_length_diff = 0
        total_original_length = 0
        total_modified_length = 0
        
        for review in reviews:
            original = review.get('original', '')
            modified = review.get('modified', '')
            if original and modified:
                # 长度变化分析
                original_len = len(original)
                modified_len = len(modified)
                length_diff = modified_len - original_len
                
                modifications.append({
                    'original_length': original_len,
                    'modified_length': modified_len,
                    'length_diff': length_diff
                })
                
                total_length_diff += length_diff
                total_original_length += original_len
                total_modified_length += modified_len
        
        # 计算平均值
        avg_length_diff = total_length_diff / len(modifications) if modifications else 0
        avg_original_length = total_original_length / len(modifications) if modifications else 0
        avg_modified_length = total_modified_length / len(modifications) if modifications else 0
        
        # 文本相似度分析（简单计算）
        import difflib
        similarity_scores = []
        for review in reviews:
            original = review.get('original', '')
            modified = review.get('modified', '')
            if original and modified:
                similarity = difflib.SequenceMatcher(None, original, modified).ratio()
                similarity_scores.append(similarity)
        
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        
        analytics = {
            'total_reviews': total_reviews,
            'average_length_difference': round(avg_length_diff, 2),
            'average_original_length': round(avg_original_length, 2),
            'average_modified_length': round(avg_modified_length, 2),
            'average_text_similarity': round(avg_similarity, 4),
            'modifications': modifications[:10]  # 只返回前10个示例
        }
        
        return jsonify({'success': True, 'data': analytics}), 200
    except Exception as e:
        print(f"Error performing analytics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """首页"""
    return jsonify({
        'message': 'Chat Review Server is running',
        'endpoints': {
            'POST /api/review': '接收审阅数据',
            'GET /api/reviews': '获取所有审阅数据',
            'GET /api/analytics': '获取数据分析结果'
        }
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)