#pragma once

#include <QGraphicsDropShadowEffect>
#include <QPropertyAnimation>
#include <QPushButton>

class AnimatedButton final : public QPushButton
{
	Q_OBJECT
	Q_PROPERTY(QSize buttonSize READ buttonSize WRITE setButtonSize)

  public:
	explicit AnimatedButton(const QString& text, QWidget* parent = nullptr);

	QSize buttonSize() const { return m_size; }
	void setButtonSize(const QSize& size);

  protected:
	void enterEvent(QEvent* event) override;
	void leaveEvent(QEvent* event) override;

  private:
	QPropertyAnimation* animation;
	QGraphicsDropShadowEffect* shadow;
	QSize m_size;
	QSize m_originalSize;
};
