#include "../include/AnimatedButton.h"

#include <QGraphicsDropShadowEffect>
#include <QTimer>

AnimatedButton::AnimatedButton(const QString& text, QWidget* parent) : QPushButton(text, parent)
{
	setStyleSheet(R"(
        QPushButton {
            background-color: #ffb3ba;
            color: #2e2e2e;
            border: 2px solid #ff6f91;
            border-radius: 12px;
            padding: 8px 16px;
        }
        QPushButton:hover {
            background-color: #ffc2d1;
        }
    )");

	shadow = new QGraphicsDropShadowEffect(this);
	shadow->setBlurRadius(20);
	shadow->setOffset(0, 5);
	shadow->setColor(QColor(0, 0, 0, 120));
	setGraphicsEffect(shadow);

	// Сохраняем оригинальный размер после первого show
	QTimer::singleShot(0, this, [this]() {
		m_originalSize = size();
		m_size = m_originalSize;
	});

	// Анимация через изменение размера
	animation = new QPropertyAnimation(this, "buttonSize");
	animation->setDuration(150);
	animation->setEasingCurve(QEasingCurve::OutCubic);
}

void AnimatedButton::setButtonSize(const QSize& size)
{
	m_size = size;
	setMinimumSize(size);
	setMaximumSize(size);
	resize(size);
}

void AnimatedButton::enterEvent(QEvent* event)
{
	if (m_originalSize.isEmpty()) {
		m_originalSize = size();
		m_size = m_originalSize;
	}

	QSize targetSize(m_originalSize.width() * 1.05, m_originalSize.height() * 1.05);
	
	animation->stop();
	animation->setStartValue(m_size);
	animation->setEndValue(targetSize);
	animation->start();

	QPushButton::enterEvent(event);
}

void AnimatedButton::leaveEvent(QEvent* event)
{
	animation->stop();
	animation->setStartValue(m_size);
	animation->setEndValue(m_originalSize);
	animation->start();

	QPushButton::leaveEvent(event);
}
