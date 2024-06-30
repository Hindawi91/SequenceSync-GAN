import os

def delete_non_py_sh_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not (file.endswith('.py') or file.endswith('.sh')):
                file_path = os.path.join(root, file)
                print(f"Deleting file: {file_path}")
                os.remove(file_path)

if __name__ == "__main__":
    directory = "/home/local/ASUAD/falhinda/NEW_FPGAN_Exp/FPGAN_Training/phase2/SequenceSync-GAN"
    delete_non_py_sh_files(directory)
