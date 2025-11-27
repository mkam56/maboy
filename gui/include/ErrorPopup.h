#pragma once

#include <QLabel>
#include <QPropertyAnimation>
#include <qdialog.h>

class ErrorPopup : public QDialog
{
	Q_OBJECT

  public:
	explicit ErrorPopup(const char* er_text, bool is_success, QWidget* parent = nullptr);
};
