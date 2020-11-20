from pdf2image import convert_from_path


def main():
    print('Hello!')
    pdf_path = '../MyNotebook.pdf'
    output_path = '../output'
    images = convert_from_path(pdf_path=pdf_path,
                               dpi=300,
                               output_folder=output_path,
                               fmt="png",                               
                               thread_count=1)

    print('Done!')


if __name__ == '__main__':
    main()
