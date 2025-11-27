#include "../include/ErrorPopup.h"

#include <QApplication>
#include <QDialog>
#include <QLabel>
#include <QMovie>
#include <QPushButton>
#include <QVBoxLayout>

ErrorPopup::ErrorPopup(const char *er_text, const bool is_success, QWidget *parent) : QDialog(parent)
{
	setWindowTitle("");
	setModal(true);
	resize(300, 250);

	auto *layout = new QVBoxLayout(this);

	auto *gifLabel = new QLabel(this);
	auto *movie = new QMovie(is_success ? ":animations/success.gif" : ":animations/error.gif");
	gifLabel->setMovie(movie);
	movie->start();

	auto *label = new QLabel(er_text, this);

	auto *closeButton = new QPushButton("To close", this);
	connect(closeButton, &QPushButton::clicked, this, &QDialog::accept);

	layout->addWidget(gifLabel, 0, Qt::AlignCenter);
	layout->addWidget(label, 0, Qt::AlignCenter);
	layout->addWidget(closeButton, 0, Qt::AlignCenter);
}
